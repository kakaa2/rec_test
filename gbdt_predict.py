import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from odps import ODPS, options
import time # 用于计时

# --- 1. 配置 ---
# ODPS 连接信息 (假设 key.py 在同一目录或 PYTHONPATH 中)
import key
# 品牌和模型信息
brand_id = 'b56508'        # 替换为你的品牌 ID
best_num_trees = 102      # 替换为你选择的最佳树数量
brand_output_dir = f'output_{brand_id}'
model_save_dir = os.path.join(brand_output_dir, 'saved_lgbm_models')
model_file_path = os.path.join(model_save_dir, f'lgbm_model_trees_{best_num_trees}.txt')

# ODPS 表和字段信息
test_table_name = 'user_pay_sample_feature_join_eval' # 你的测试数据表名
# **重要**: 调整下面的 WHERE 子句以选择你想要预测的数据范围
# 例如，如果你想预测 20130917 到 20130930 的数据
test_data_filter = "ds='20130923'" # ***根据需要修改日期范围***

# 批处理设置
num_batches = 10           # 将数据分成多少批处理，根据内存调整

# Top-K 计算设置
top_k_values = [1000, 3000, 5000, 10000, 50000]

# 特征和标签列定义 (应与训练时一致)
target_column = 'label'
# 'rnd' 是用于分批的，不需要作为特征，也不应在 exclude_columns 中（除非训练时也不包含）
exclude_columns = ['user_id', 'brand_id', 'bizdate', 'rnd', 'ds'] # 确保与训练脚本一致

# --- 2. 初始化 ODPS 和加载模型 ---

print("--- 初始化 ODPS 连接 ---")
try:
    o = ODPS(
        key.access_id,
        key.access_key,
        key.project_name,
        key.end_point
    )
    options.tunnel.use_instance_tunnel = True
    options.tunnel.limit_instance_tunnel = False
    print("ODPS 连接成功。")
except Exception as e:
    print(f"ODPS 连接失败: {e}")
    exit()

print(f"\n--- 加载 LightGBM 模型 ---")
if not os.path.exists(model_file_path):
    print(f"错误：找不到模型文件: {model_file_path}")
    exit()

try:
    bst_loaded = lgb.Booster(model_file=model_file_path)
    print(f"成功加载模型: {model_file_path}")
    # 获取模型训练时使用的特征名 (以防万一需要检查)
    model_feature_names = bst_loaded.feature_name()
    print(f"模型需要 {len(model_feature_names)} 个特征。")
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit()

# --- 3. 分批预测 ---

print(f"\n--- 开始分批预测 (共 {num_batches} 批) ---")
all_results_list = []
total_rows_processed = 0
start_time_prediction = time.time()

for i in range(num_batches):
    batch_start_time = time.time()
    lower_bound = float(i) / num_batches
    upper_bound = float(i + 1) / num_batches

    # 构建当前批次的 SQL 查询
    # 注意最后一个批次包含上界 (rnd <= 1.0)
    rnd_filter = f"rnd >= {lower_bound} AND rnd < {upper_bound}"
    if i == num_batches - 1:
        rnd_filter = f"rnd >= {lower_bound} AND rnd <= 1.0" # 包含 1.0

    sql_batch = f"""
    SELECT *
    FROM {test_table_name}
    WHERE brand_id='{brand_id}' AND ({test_data_filter}) AND ({rnd_filter});
    """
    print(f"SQL 查询: {sql_batch}")
    # WHERE brand_id='{brand_id}' AND ds>='20130917'and ds<='20130930'AND (rnd >= 0.0 AND rnd < 0.1);

    print(f"\n批次 {i+1}/{num_batches} (rnd范围: [{lower_bound:.2f}, {'<' if i < num_batches - 1 else '<='}{upper_bound:.2f}))")

    try:
        query_job = o.execute_sql(sql_batch)
        result = query_job.open_reader(tunnel=True)
        # 可以指定 use_cols 来只读取需要的列，减少内存占用
        # cols_to_read = exclude_columns + [target_column] + model_feature_names # 理论上需要这些
        # df_batch = result.to_pandas(n_process=4, use_cols=cols_to_read)
        df_batch = result.to_pandas(n_process=os.cpu_count()) # 先读取全部，再筛选
        print(f"批次 {i+1} 数据加载完成，形状: {df_batch.shape}")

    except Exception as e:
        print(f"批次 {i+1} 从 ODPS 读取数据时出错: {e}")
        continue # 跳过这个批次或根据需要处理错误

    if df_batch.empty:
        print(f"批次 {i+1} 数据为空，跳过。")
        continue

    total_rows_processed += len(df_batch)

    # 获取当前批次的特征列 (与训练时一致的方式)
    batch_feature_columns = [col for col in df_batch.columns if col not in exclude_columns and col != target_column]

    # 检查特征是否匹配 (可选但推荐)
    if set(batch_feature_columns) != set(model_feature_names):
         print(f"警告：批次 {i+1} 的特征列与模型训练时的特征列不完全匹配！")
         print(f"批次特征数: {len(batch_feature_columns)}, 模型特征数: {len(model_feature_names)}")
         # 尝试按模型需要的特征顺序排列和选取
         try:
            X_batch = df_batch[model_feature_names] # 按模型顺序选择
            print("已尝试按模型特征顺序重新选择特征。")
         except KeyError as e:
             print(f"错误：无法从批次数据中找到模型需要的特征: {e}。跳过此批次。")
             continue
    else:
        # 特征匹配，直接选择
        X_batch = df_batch[batch_feature_columns]

    # 进行预测
    try:
        batch_preds_proba = bst_loaded.predict(X_batch, num_iteration=best_num_trees)
    except Exception as e:
        print(f"批次 {i+1} 预测时出错: {e}")
        continue

    # 准备结果 DataFrame (只保留标签和预测分数)
    batch_results = pd.DataFrame({
        'label': df_batch[target_column],
        'prediction_score': batch_preds_proba
        # 可以添加 'user_id': df_batch['user_id'] 等用于后续分析
    })

    all_results_list.append(batch_results)
    batch_end_time = time.time()
    print(f"批次 {i+1} 处理完成，耗时: {batch_end_time - batch_start_time:.2f} 秒")

end_time_prediction = time.time()
print(f"\n--- 所有批次预测完成 ---")
print(f"总处理行数: {total_rows_processed}")
print(f"总预测耗时: {end_time_prediction - start_time_prediction:.2f} 秒")

# --- 4. 合并结果并计算 Top-K Recall ---

if not all_results_list:
    print("\n错误：没有成功处理任何批次的数据，无法进行后续计算。")
    exit()

print("\n--- 合并所有预测结果 ---")
start_time_calc = time.time()
df_all_results = pd.concat(all_results_list, ignore_index=True)
print(f"合并后总结果数: {len(df_all_results)}")

# 计算测试集中总的正样本数
total_positives = df_all_results['label'].sum()
print(f"测试数据集中总的正样本 (label=1) 数量: {total_positives}")

if total_positives == 0:
    print("\n警告：测试数据集中没有正样本，无法计算 Recall。")
    exit()

print("\n--- 计算 Top-K Recall ---")
# 按预测分数降序排序
print("按预测分数排序...")
df_sorted = df_all_results.sort_values('prediction_score', ascending=False).reset_index(drop=True)
print("排序完成。")

recall_results = {}

for k in top_k_values:
    if k > len(df_sorted):
        print(f"警告: Top {k} 超出总预测数量 ({len(df_sorted)})，将使用所有预测结果计算。")
        k_actual = len(df_sorted)
    else:
        k_actual = k
    # 获取 Top-K 的预测结果
    top_k_df = df_sorted.head(k_actual)
    # 计算 Top-K 中正样本的数量
    positives_in_top_k = top_k_df['label'].sum()
    # 计算 Recall
    recall = positives_in_top_k / total_positives

    recall_results[k] = recall
    print(f"Top {k_actual}:")
    print(f"  - 包含的正样本数: {positives_in_top_k}")
    print(f"  - Recall (占总正样本比例): {recall:.6f}")

end_time_calc = time.time()
print(f"\n--- 计算完成 ---")
print(f"结果合并与计算耗时: {end_time_calc - start_time_calc:.2f} 秒")

recall_df = pd.DataFrame(list(recall_results.items()), columns=['TopK', 'Recall'])
recall_output_file = os.path.join(brand_output_dir, 'prediction_recall_report.csv')
try:
    recall_df.to_csv(recall_output_file, index=False)
    print(f"\nRecall 结果已保存到: {recall_output_file}")
except IOError as e:
    print(f"\n错误：无法保存 Recall 结果文件: {e}")

# 如果需要保存带分数的完整排序列表 (可能非常大)
# sorted_output_file = os.path.join(brand_output_dir, 'predictions_sorted.csv.gz') # 使用压缩
# try:
#     print(f"\n正在保存排序后的完整预测结果到: {sorted_output_file} (可能需要较长时间)...")
#     df_sorted[['label', 'prediction_score']].to_csv(sorted_output_file, index=False, compression='gzip')
#     print("完整预测结果保存完成。")
# except IOError as e:
#      print(f"\n错误：无法保存完整预测结果文件: {e}")
print("\n脚本执行完毕.")