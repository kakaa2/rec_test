import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os # 用于创建文件夹
from odps import ODPS, options
import key


# 初始化 ODPS 入口
o = ODPS(
    key.access_id,
    key.access_key,
    key.project_name,
    key.end_point
)

# 全局启用 Instance Tunnel 并关闭 limit 限制
options.tunnel.use_instance_tunnel = True
options.tunnel.limit_instance_tunnel = False

brand_id='b47686'
sql ='''
SELECT *
FROM user_pay_sample_feature_join
where ds>='20130701'and ds<='20130916'and brand_id='{brand}' limit 1000;
'''.format(brand=brand_id)

print(sql)
query_job =o.execute_sql(sql)
result = query_job.open_reader(tunnel=True)
df = result.to_pandas(n_process=4) 
print('read data finish')

print(f"数据加载完成，形状: {df.shape}")

# 定义标签列和需要排除的列
target_column = 'label'
exclude_columns = ['user_id', 'brand_id', 'bizdate', 'rnd', 'ds'] # 根据你的实际情况调整

# 获取特征列
feature_columns = [col for col in df.columns if col not in exclude_columns and col != target_column]

# 分离特征和标签
X = df[feature_columns]
y = df[target_column]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape}, {y_train.shape}")
print(f"测试集大小: {X_test.shape}, {y_test.shape}")
print(f"使用的特征数量: {len(feature_columns)}")

# --- 2. LightGBM 核心 API 准备 ---

# 将数据转换为 LightGBM Dataset 格式，这是核心 API 的要求
# free_raw_data=False 如果后续还需要原始数据，设为False，否则设为True以节省内存
lgb_train = lgb.Dataset(X_train, y_train, feature_name=feature_columns, free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, feature_name=feature_columns, free_raw_data=False)

# 定义模型参数 (不包含迭代次数 n_estimators/num_boost_round)
# scale_pos_weight 用于处理类别不平衡
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1
print(f"计算得到的 scale_pos_weight: {scale_pos_weight_value:.2f}")

params = {
    'objective': 'binary',          # 二分类任务
    'metric': 'auc',                # 评估指标
    'boosting_type': 'gbdt',
    'num_leaves': 31,               # 叶子节点数
    'learning_rate': 0.05,          # 学习率
    'feature_fraction': 0.9,        # 建树的特征选择比例 (类似 colsample_bytree)
    'bagging_fraction': 0.8,        # 数据采样比例 (类似 subsample)
    'bagging_freq': 5,              # 每 5 次迭代执行一次 bagging
    'verbose': -1,                  # 控制日志输出级别，-1 表示静默
    'n_jobs': -1,                   # 使用所有CPU核心
    'seed': 42,                     # 随机种子
    'scale_pos_weight': scale_pos_weight_value, # 处理类别不平衡
}

# --- 3. 迭代训练、评估和保存 ---

max_trees = 100  # 最大树数
interval = 5     # 评估和保存的间隔
model_save_dir = 'saved_lgbm_models' # 模型保存的文件夹
auc_log_file = 'auc_log.txt'         # AUC 记录文件

# 创建保存模型的文件夹 (如果不存在)
os.makedirs(model_save_dir, exist_ok=True)

# 用于存储 Booster 对象 (训练过程中的模型状态)
current_booster = None
auc_results = [] # 存储 (树数, 训练集AUC, 测试集AUC)

print("\n开始迭代训练、评估和保存...")

# 使用 lgb.train 进行迭代训练
# 循环从 interval 到 max_trees，步长为 interval
for n_trees in range(interval, max_trees + 1, interval):
    print(f"\n--- 训练到 {n_trees} 棵树 ---")

    # 计算本次需要额外训练的轮数（树的数量）
    num_boost_round_step = interval

    # 进行增量训练
    # init_model=current_booster: 从上一次的状态继续训练
    # keep_training_booster=True: 允许 lgb.train 修改并返回 booster 对象
    current_booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=num_boost_round_step, # 本次迭代增加的树的数量
        init_model=current_booster,          # 从之前的模型继续
        keep_training_booster=True           # 关键：允许增量训练
        # valid_sets=[lgb_train, lgb_eval], # 可以在训练时监控，但我们后面会手动计算
        # verbose_eval=False # 不在训练过程中打印评估信息
    )

    # 评估当前模型 (已有 n_trees 棵树)
    # 预测概率
    y_train_pred_proba = current_booster.predict(X_train, num_iteration=n_trees)
    y_test_pred_proba = current_booster.predict(X_test, num_iteration=n_trees)

    # 计算 AUC
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)

    # 记录 AUC
    auc_results.append((n_trees, train_auc, test_auc))
    print(f"树数量: {n_trees}, 训练集 AUC: {train_auc:.6f}, 测试集 AUC: {test_auc:.6f}")

    # 保存当前模型状态 (包含 n_trees 棵树)
    model_filename = os.path.join(model_save_dir, f'lgbm_model_trees_{n_trees}.txt')
    current_booster.save_model(model_filename, num_iteration=n_trees)
    print(f"模型已保存到: {model_filename}")

print("\n--- 所有迭代完成 ---")

# --- 4. 将 AUC 结果写入文件 ---
try:
    with open(auc_log_file, 'w', encoding='utf-8') as f:
        f.write("棵树数量,训练集AUC,测试集AUC\n") # 写入表头
        for trees, train_auc, test_auc in auc_results:
            f.write(f"{trees},{train_auc:.8f},{test_auc:.8f}\n")
    print(f"\nAUC 结果已成功写入到文件: {auc_log_file}")
except IOError as e:
    print(f"\n错误：无法写入 AUC 日志文件 '{auc_log_file}': {e}")

print("\n脚本执行完毕.")

# --- (可选) 如何加载保存的模型 ---
# later_trees = 30 # 比如想加载训练了30棵树的模型
# loaded_model_file = os.path.join(model_save_dir, f'lgbm_model_trees_{later_trees}.txt')
# if os.path.exists(loaded_model_file):
#     bst_loaded = lgb.Booster(model_file=loaded_model_file)
#     print(f"\n成功加载模型: {loaded_model_file}")
#     # 可以用 bst_loaded 进行预测
#     # y_pred_proba_loaded = bst_loaded.predict(X_test)
#     # loaded_auc = roc_auc_score(y_test, y_pred_proba_loaded)
#     # print(f"加载的模型在测试集上的 AUC: {loaded_auc:.6f}")
# else:
#     print(f"\n错误：找不到要加载的模型文件: {loaded_model_file}")