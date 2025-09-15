import rasterio
import numpy as np
import os

# --------------------------
# 配置参数（根据实际情况修改）
# --------------------------
mask_path = "/home/xyf/Downloads/landuse/datas/mask/mask.tif"  # mask文件路径
input_tif_paths = [     # 待处理的多年份TIFF路径（与之前的年份数据对应）
    "/home/xyf/Downloads/landuse/datas/landuse/tif_to20/08_to2020.tif",
    "/home/xyf/Downloads/landuse/datas/landuse/tif_to20/12_to2020.tif",
    "/home/xyf/Downloads/landuse/datas/landuse/tif_to20/16_to2020.tif",
    "/home/xyf/Downloads/landuse/datas/landuse/tif_to20/20_to2020.tif"
]
output_dir = "masked_tifs"  # 处理后的数据保存目录
mask_valid_value = 1        # mask中表示"有效区域"的值（通常是1）
nodata_value = 0          # 处理后无效区域的填充值（与之前的二值化代码保持一致）

# --------------------------
# 核心处理函数
# --------------------------
def read_mask(mask_path):
    """读取mask文件并返回掩膜矩阵和元数据"""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # 读取mask的第一个波段
        mask_meta = src.meta.copy()
        # 确保mask是二值化的（仅保留有效/无效标记）
        mask = (mask == mask_valid_value).astype(np.uint8)  # 有效区域为1，无效为0
    return mask, mask_meta

def apply_mask_to_tif(input_tif_path, mask, nodata):
    """将mask应用到单幅TIFF数据，返回处理后的数据和原数据元信息"""
    with rasterio.open(input_tif_path) as src:
        data = src.read(1)  # 读取数据的第一个波段
        meta = src.meta.copy()  # 保留原数据的地理信息等元数据
        
        # 应用掩膜：mask为0的区域设为无数据值，mask为1的区域保留原数据
        masked_data = np.where(mask == 1, data, nodata)
        
        # 如果原数据是二值化的（0和1），确保处理后仍保持二值特性
        if meta['dtype'] == 'uint8':
            masked_data = masked_data.astype(np.uint8)
    
    return masked_data, meta

def process_all_tifs_with_mask(mask_path, input_paths, output_dir, nodata):
    """批量处理所有TIFF文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取mask
    print(f"读取mask文件: {mask_path}")
    mask, mask_meta = read_mask(mask_path)
    mask_height, mask_width = mask.shape
    
    # 逐个处理输入TIFF
    for i, tif_path in enumerate(input_paths):
        tif_name = os.path.basename(tif_path)
        print(f"处理第{i+1}个文件: {tif_name}")
        
        # 应用掩膜
        masked_data, meta = apply_mask_to_tif(tif_path, mask, nodata)
        
        # 检查尺寸是否匹配（防止mask与TIFF尺寸不一致）
        data_height, data_width = masked_data.shape
        if (data_height, data_width) != (mask_height, mask_width):
            print(f"警告：{tif_name}与mask尺寸不匹配（{data_height}x{data_width} vs {mask_height}x{mask_width}），已跳过该文件")
            continue
        
        # 保存处理后的文件
        output_path = os.path.join(output_dir, f"masked_{tif_name}")
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(masked_data, 1)  # 写入第一个波段
        print(f"处理完成，保存至: {output_path}")
    
    print("\n所有文件处理完毕！")

# --------------------------
# 执行处理
# --------------------------
if __name__ == "__main__":
    process_all_tifs_with_mask(
        mask_path=mask_path,
        input_paths=input_tif_paths,
        output_dir=output_dir,
        nodata=nodata_value
    )
