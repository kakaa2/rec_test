# 确保你在激活的环境中：
conda activate pytorch-2.1.2

# 找到 Conda 环境中的 libstdc++.so.6 文件路径
CONDA_LIBSTDC_PATH=$(find ~/.conda/envs/pytorch-2.1.2/ -name "libstdc++.so.6" | grep '/lib/' | head -n 1)

# 检查是否找到了路径
echo "找到 Conda 的 libstdc++ 路径: $CONDA_LIBSTDC_PATH"

# 使用 LD_PRELOAD 运行你的脚本 (如果上面的路径找到了)
if [ -n "$CONDA_LIBSTDC_PATH" ]; then
    LD_PRELOAD=$CONDA_LIBSTDC_PATH python gbdt_train.py
else
    echo "错误：未能自动找到 Conda 环境中的 libstdc++.so.6 路径。"
    # 你可以手动查找并设置路径:
    # export LD_PRELOAD=/root/.conda/envs/pytorch-2.1.2/lib/libstdc++.so.6 
    # python gbdt_train.py
fi