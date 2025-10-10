import rasterio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv2D, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --------------------------
# 1. 配置参数（修复无数据值类型问题）
# --------------------------
tiff_paths = [
    "datas/landuse/masked_tifs/masked_08_to2020.tif",   # 第1年二值化数据
    "datas/landuse/masked_tifs/masked_12_to2020.tif",   # 第2年二值化数据
    "datas/landuse/masked_tifs/masked_16_to2020.tif"    # 第3年二值化数据
]
actual_t4_path = "datas/landuse/masked_tifs/masked_08_to2020.tif"  # 第四年实际数据
predicted_t4_path = "/home/xyf/Downloads/landuse/convLSTM/scipts/binary_results/predicted_t4_binary.tif"  # 预测结果
output_dir = "/home/xyf/Downloads/landuse/convLSTM/scipts//binary_results"                  # 结果目录
nodata_value = 255                             # 改为uint8支持的无数据值（替换-9999）
epochs = 50                                    # 训练轮次

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# --------------------------
# 2. 数据读取与二值化处理（修复维度问题）
# --------------------------
def read_tiff(file_path):
    """读取TIFF文件并确保数据格式正确"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            meta = src.meta.copy()
            meta['dtype'] = 'uint8'  # 明确使用uint8存储0/1和无数据值
            height, width = data.shape
        return data, meta, height, width
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {str(e)}")
        raise

def binarize_data(data, original_nodata=-9999):
    """将数据二值化（0和1），并处理无数据值"""
    # 识别原始无数据值（如-9999）
    mask = (data != original_nodata)
    
    # 二值化：大于0.5的值设为1，否则为0
    data_binary = np.zeros_like(data, dtype=np.uint8)
    data_binary[mask & (data > 0.5)] = 1
    
    # 用新的无数据值标记（255，uint8支持）
    data_binary[~mask] = nodata_value
    
    return data_binary, mask

# 读取并处理前三年数据
print("===== 读取并二值化训练数据 =====")
tiff_data = []
meta = None
height = None
width = None
valid_mask_2d = None  # 2维空间掩码（height, width）

for path in tiff_paths:
    print(f"处理数据: {path}")
    data, meta, h, w = read_tiff(path)
    
    # 二值化处理（注意：这里指定原始无数据值为-9999）
    data_binary, mask = binarize_data(data, original_nodata=-9999)
    tiff_data.append(data_binary)
    
    # 记录有效数据掩码（空间维度）
    if valid_mask_2d is None:
        valid_mask_2d = mask
        height, width = h, w
    else:
        valid_mask_2d &= mask  # 取空间交集
        assert (h, w) == (height, width), f"数据尺寸不一致: {path} 为 {h}x{w}"

# 转换为numpy数组 (时间步, 高度, 宽度)
tiff_array = np.stack(tiff_data, axis=0)  # 形状: (3, height, width)

# 关键修复：扩展掩码到时间维度（3, height, width）
valid_mask = np.repeat(valid_mask_2d[np.newaxis, ...], tiff_array.shape[0], axis=0)

# 验证二值化结果
print(f"数据二值化完成，形状: {tiff_array.shape}")
print(f"二值化验证：数据值为 {np.unique(tiff_array[valid_mask])}")  # 现在维度匹配

# 转换为模型输入格式 (样本数, 时间步, 高度, 宽度, 通道数)
X = tiff_array[np.newaxis, ..., np.newaxis]  # (1, 3, h, w, 1)

# --------------------------
# 3. 模型构建与训练
# --------------------------
print("\n===== 构建并训练二值分类模型 =====")
model = Sequential([
    ConvLSTM2D(
        filters=16,
        kernel_size=(3, 3),
        input_shape=(None, height, width, 1),
        padding='same',
        return_sequences=False,
        activation='relu'
    ),
    BatchNormalization(),
    Dropout(0.3),
    Conv2D(
        filters=1,
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same'
    )
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 准备训练数据
X_train = X[:, :2, ...]  # 前2年
y_train = X[:, 2:3, ...] # 第3年作为目标
y_train = np.squeeze(y_train, axis=1)

# 训练模型
history = model.fit(
    X_train, y_train,
    batch_size=1,
    epochs=epochs,
    verbose=1
)

# 保存训练曲线
os.makedirs(output_dir, exist_ok=True)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('loss --- epochs')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('accuracy --- epochs')
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "训练曲线.png"), dpi=300)
plt.close()

# --------------------------
# 4. 预测并处理结果
# --------------------------
print("\n===== 预测第四年数据 =====")
X_pred = X
y_pred_prob = model.predict(X_pred)

# 二值化预测结果
def postprocess_binary_prediction(pred_prob, mask_2d, nodata):
    pred_prob = np.squeeze(pred_prob)
    pred_binary = (pred_prob > 0.5).astype(np.uint8)
    pred_binary[~mask_2d] = nodata  # 使用2维空间掩码
    return pred_binary

predicted_data = postprocess_binary_prediction(
    y_pred_prob, valid_mask_2d, nodata_value
)

# 保存预测结果
with rasterio.open(predicted_t4_path, 'w', **meta) as dst:
    dst.write(predicted_data, 1)
print(f"二值化预测结果已保存至: {os.path.abspath(predicted_t4_path)}")
print(f"预测结果验证：值为 {np.unique(predicted_data[valid_mask_2d])}")

# --------------------------
# 5. 评估预测结果
# --------------------------
print("\n===== 评估预测准确率 =====")
# 读取并处理实际数据
actual_data, _, _, _ = read_tiff(actual_t4_path)
actual_data, _ = binarize_data(actual_data, original_nodata=-9999)

assert predicted_data.shape == actual_data.shape, "尺寸不匹配"

# 提取有效数据
eval_mask = valid_mask_2d & (actual_data != nodata_value) & (predicted_data != nodata_value)
pred_valid = predicted_data[eval_mask].flatten()
actual_valid = actual_data[eval_mask].flatten()

print(f"有效评估数据点数量: {len(pred_valid)}")
if len(pred_valid) == 0:
    raise ValueError("未找到有效评估数据点")

# 计算分类指标
metrics = {
    "准确率 (Accuracy)": accuracy_score(actual_valid, pred_valid),
    "精确率 (Precision)": precision_score(actual_valid, pred_valid),
    "召回率 (Recall)": recall_score(actual_valid, pred_valid),
    "F1分数 (F1-Score)": f1_score(actual_valid, pred_valid)
}

# 计算FOM（优度系数）：适用于变化检测/目标区域检测
# FOM = 正确检测的目标像素 / (正确检测 + 漏检 + 虚检)
cm = confusion_matrix(actual_valid, pred_valid)
if cm.shape == (2, 2):
    TN, FP, FN, TP = cm.ravel()  # 混淆矩阵拆解：TN[0,0], FP[0,1], FN[1,0], TP[1,1]
    # 防止分母为0
    denominator = TP + FN + FP
    fom = TP / denominator if denominator != 0 else 0.0
    metrics["优度系数 (FOM)"] = fom
else:
    # 处理单类别情况（所有样本都是0或1）
    metrics["优度系数 (FOM)"] = np.nan

# 打印指标
print("\n===== 二值分类评估指标 =====")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# 保存指标
with open(os.path.join(output_dir, "分类评估指标.txt"), "w", encoding="utf-8") as f:
    f.write("===== 二值分类评估指标 =====\n")
    for name, value in metrics.items():
        f.write(f"{name}: {value:.4f}\n")

# --------------------------
# 6. 可视化结果
# --------------------------
print("\n===== 生成可视化结果 =====")

# 错误分类区域
error_mask = eval_mask & (predicted_data != actual_data)
correct_mask = eval_mask & (predicted_data == actual_data)

# 6.1 对比图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(actual_data, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Actual Binary Data")
axes[0].axis("off")

axes[1].imshow(predicted_data, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Predicted Binary Data")
axes[1].axis("off")

# 错误区域标记
base = np.zeros((height, width, 3), dtype=np.uint8)
base[correct_mask] = [255, 255, 255]
base[error_mask] = [255, 0, 0]
axes[2].imshow(base)
axes[2].set_title(f"error field (red) - error_rate: {1-metrics['准确率 (Accuracy)']:.2%}")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "二值化对比图.png"), dpi=300)
plt.close()

# 6.2 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(actual_valid, pred_valid)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['predict_0', 'predict_1'],
            yticklabels=['truth_0', 'truth_1'])
plt.xlabel('preditction class')
plt.ylabel('truth class')
plt.title('metrics confusion matrix')
plt.savefig(os.path.join(output_dir, "混淆矩阵.png"), dpi=300)
plt.close()

print(f"所有结果已保存至: {os.path.abspath(output_dir)}")
print("\n===== 流程完成 =====")
