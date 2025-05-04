import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os # 用于创建文件夹
from odps import ODPS, options
# 假设 key.py 在同一目录下或 PYTHONPATH 中
try:
    import key
except ImportError:
    print("错误：无法导入 key.py 文件。请确保该文件存在并包含必要的 ODPS 凭证。")
    # 你可以在这里提供默认值或直接退出脚本
    # 例如:
    # key = type('obj', (object,), {'access_id':'YOUR_ID', 'access_key':'YOUR_KEY', 'project_name':'YOUR_PROJECT', 'end_point':'YOUR_ENDPOINT'})()
    # 或者
    exit()


# 初始化 ODPS 入口
try:
    o = ODPS(
        key.access_id,
        key.access_key,
        key.project_name,
        key.end_point
    )
    # 全局启用 Instance Tunnel 并关闭 limit 限制
    options.tunnel.use_instance_tunnel = True
    options.tunnel.limit_instance_tunnel = False
    print("ODPS 连接成功。")
except Exception as e:
    print(f"ODPS 连接失败: {e}")
    exit()
# b47686:韩都衣舍 b56508:三星手机 b62063:诺基亚 b78739:LILY
brand_ids=['b47686','b56508','b62063','b78739']
brand_id=brand_ids[1] # <--- 品牌 ID 在这里定义

# --- 改动开始 ---
# 1. 定义品牌专属的根输出目录
brand_output_dir = f'output_{brand_id}' # 例如 'output_b47686'

# 2. 定义模型保存目录 (在品牌专属目录下)
model_save_dir = os.path.join(brand_output_dir, 'saved_lgbm_models') # 例如 'output_b47686/saved_lgbm_models'

# 3. 定义 AUC 日志文件路径 (在品牌专属目录下)
auc_log_file = os.path.join(brand_output_dir, 'auc_log.txt')         # 例如 'output_b47686/auc_log.txt'

# 4. 创建保存模型的文件夹 (包括品牌专属目录，如果不存在)
#    os.makedirs 会创建所有必需的中间目录
os.makedirs(model_save_dir, exist_ok=True)
print(f"输出将保存在品牌专属目录: {brand_output_dir}")
# --- 改动结束 ---


sql ='''
SELECT *
FROM user_pay_sample_feature_join
where ds>='20130701'and ds<='20130916'and brand_id='{brand}' ;
'''.format(brand=brand_id)

print(sql)
try:
    query_job =o.execute_sql(sql)
    result = query_job.open_reader(tunnel=True)
    df = result.to_pandas(n_process=4)
    print('read data finish')
except Exception as e:
    print(f"从 ODPS 读取数据时出错: {e}")
    exit()


print(f"数据加载完成，形状: {df.shape}")
if df.empty:
    print("错误：加载的数据为空，无法继续训练。请检查 SQL 查询或源数据。")
    exit()

# 定义标签列和需要排除的列
target_column = 'label'
exclude_columns = ['user_id', 'brand_id', 'bizdate', 'rnd', 'ds'] # 根据你的实际情况调整

# 获取特征列
feature_columns = [col for col in df.columns if col not in exclude_columns and col != target_column]

# 分离特征和标签
X = df[feature_columns]
y = df[target_column]

# 检查标签列是否存在且非空
if target_column not in df.columns:
    print(f"错误：标签列 '{target_column}' 不在数据中。")
    exit()
if y.isnull().any():
    print(f"警告：标签列 '{target_column}' 包含缺失值。")
    # 可以选择填充或删除包含缺失标签的行
    # df = df.dropna(subset=[target_column])
    # X = df[feature_columns]
    # y = df[target_column]
    # print(f"处理缺失标签后，数据形状: {df.shape}")


# 检查标签分布
print(f"标签分布:\n{y.value_counts(normalize=True)}")
if len(y.unique()) < 2:
     print(f"错误：标签列 '{target_column}' 只包含一个类别，无法进行二分类训练。")
     exit()


# 划分训练集和测试集
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # stratify 需要至少两个类别
    )
except ValueError as e:
     print(f"划分数据集时出错: {e}")
     print("这通常发生在某个类别的样本数过少，无法进行分层抽样。请检查标签分布。")
     # 可以考虑不使用 stratify，但这可能导致训练/测试集分布偏差
     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     exit()


print(f"训练集大小: {X_train.shape}, {y_train.shape}")
print(f"测试集大小: {X_test.shape}, {y_test.shape}")
print(f"使用的特征数量: {len(feature_columns)}")

# --- 2. LightGBM 核心 API 准备 ---

# 将数据转换为 LightGBM Dataset 格式
lgb_train = lgb.Dataset(X_train, y_train, feature_name=feature_columns, free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, feature_name=feature_columns, free_raw_data=False)

# 定义模型参数
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count

# 检查 pos_count 是否大于 0，避免除以零错误
if pos_count > 0:
    scale_pos_weight_value = neg_count / pos_count
else:
    scale_pos_weight_value = 1  # 如果没有正样本，设为1（或根据情况处理）
    print("警告：训练集中没有正样本 (label=1)。scale_pos_weight 设置为 1。")

print(f"计算得到的 scale_pos_weight: {scale_pos_weight_value:.2f}")

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'scale_pos_weight': scale_pos_weight_value,
}

# --- 3. 迭代训练、评估和保存 ---

max_trees = 300
interval = 3
# model_save_dir 和 auc_log_file 已在前面根据 brand_id 定义

current_booster = None
auc_results = []

print("\n开始迭代训练、评估和保存...")

for n_trees in range(interval, max_trees + 1, interval):
    print(f"\n--- 训练到 {n_trees} 棵树 ---")

    num_boost_round_step = interval

    current_booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=num_boost_round_step,
        init_model=current_booster,
        keep_training_booster=True
    )

    # 评估当前模型
    y_train_pred_proba = current_booster.predict(X_train, num_iteration=n_trees)
    y_test_pred_proba = current_booster.predict(X_test, num_iteration=n_trees)

    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)

    auc_results.append((n_trees, train_auc, test_auc))
    print(f"树数量: {n_trees}, 训练集 AUC: {train_auc:.6f}, 测试集 AUC: {test_auc:.6f}")

    # 保存当前模型状态 (路径已包含 brand_id)
    # 注意：这里的 model_save_dir 已经是品牌专属路径了
    model_filename = os.path.join(model_save_dir, f'lgbm_model_trees_{n_trees}.txt')
    current_booster.save_model(model_filename, num_iteration=n_trees)
    print(f"模型已保存到: {model_filename}")

print("\n--- 所有迭代完成 ---")

# --- 4. 将 AUC 结果写入文件 ---
try:
    # 注意：这里的 auc_log_file 已经是品牌专属路径了
    with open(auc_log_file, 'w', encoding='utf-8') as f:
        f.write("棵树数量,训练集AUC,测试集AUC\n")
        for trees, train_auc, test_auc in auc_results:
            f.write(f"{trees},{train_auc:.8f},{test_auc:.8f}\n")
    print(f"\nAUC 结果已成功写入到文件: {auc_log_file}")
except IOError as e:
    print(f"\n错误：无法写入 AUC 日志文件 '{auc_log_file}': {e}")

print("\n脚本执行完毕.")

# --- (可选) 如何加载保存的模型 (路径也需要包含 brand_id) ---
# later_trees = 30
# loaded_model_file = os.path.join(model_save_dir, f'lgbm_model_trees_{later_trees}.txt') # model_save_dir 已是品牌专属
# if os.path.exists(loaded_model_file):
#     bst_loaded = lgb.Booster(model_file=loaded_model_file)
#     print(f"\n成功加载模型: {loaded_model_file}")
#     # y_pred_proba_loaded = bst_loaded.predict(X_test)
#     # loaded_auc = roc_auc_score(y_test, y_pred_proba_loaded)
#     # print(f"加载的模型在测试集上的 AUC: {loaded_auc:.6f}")
# else:
#     print(f"\n错误：找不到要加载的模型文件: {loaded_model_file}")