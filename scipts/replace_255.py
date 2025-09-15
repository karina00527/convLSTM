import rasterio
import numpy as np

def replace_255_with_0(input_tif, output_tif):
    # 打开 2020 年的图像文件
    with rasterio.open(input_tif) as src:
        # 读取图像数据
        data = src.read(1)  # 假设图像只有一个波段
        # 将值为 255 的像素替换为 0
        data[data == 255] = 0

        # 创建新的图像文件来保存结果
        profile = src.profile
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(data, 1)  # 写入第一个波段

    print(f"✅ 已将 255 值替换为 0，并保存为 {output_tif}")

# 输入和输出文件路径
input_tif = "datas/landuse/tif_only/2020.tif"  # 2020年的原始图像路径
output_tif = "datas/landuse/tif_only/2020_1.tif"  # 修改后的图像路径

# 调用函数进行处理
replace_255_with_0(input_tif, output_tif)
