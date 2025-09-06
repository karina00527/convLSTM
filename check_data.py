# 检查数据是否对齐
import os
import rasterio
import numpy as np
from typing import Dict, List

def check_raster_consistency(reference_path: str, files_to_check: List[str]) -> None:
    """检查所有栅格数据是否与参考数据具有相同的空间属性"""
    
    # 读取参考数据
    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile
        ref_bounds = ref.bounds
        print(f"\n参考数据 ({os.path.basename(reference_path)}) 信息:")
        print(f"- 尺寸: {ref.height} x {ref.width}")
        print(f"- 空间范围: {ref_bounds}")
        print(f"- 投影: {ref.crs}")
        print(f"- 分辨率: {ref.res}")
        
    print("\n检查其他文件...")
    all_consistent = True
    
    for path in files_to_check:
        try:
            with rasterio.open(path) as src:
                filename = os.path.basename(path)
                is_consistent = True
                errors = []
                
                # 检查尺寸
                if (src.height, src.width) != (ref_profile['height'], ref_profile['width']):
                    is_consistent = False
                    errors.append(f"尺寸不匹配: {src.height}x{src.width}")
                
                # 检查投影
                if src.crs != ref_profile['crs']:
                    is_consistent = False
                    errors.append(f"投影不匹配: {src.crs}")
                
                # 检查分辨率
                if src.res != ref_profile['transform'].a:
                    is_consistent = False
                    errors.append(f"分辨率不匹配: {src.res}")
                
                # 检查空间范围
                if not np.allclose(src.bounds, ref_bounds, rtol=1e-5):
                    is_consistent = False
                    errors.append(f"空间范围不匹配: {src.bounds}")
                
                # 输出结果
                status = "✅" if is_consistent else "❌"
                print(f"\n{status} {filename}")
                if not is_consistent:
                    all_consistent = False
                    for err in errors:
                        print(f"  - {err}")
                        
        except Exception as e:
            print(f"\n❌ 无法读取 {os.path.basename(path)}")
            print(f"  - 错误: {str(e)}")
            all_consistent = False
    
    print("\n=== 检查结果 ===")
    if all_consistent:
        print("✅ 所有文件空间属性一致")
    else:
        print("❌ 存在不一致的文件，请检查上述错误信息")

if __name__ == "__main__":
    # 设置参考文件（2020年数据作为参考）
    reference = r"D:\paper2DATA\inputdata\landuse\2020.tif"
    
    # 需要检查的文件列表
    files_to_check = [
        # 土地利用数据
        r"D:\paper2DATA\inputdata\landuse\2008_to2020.tif",
        r"D:\paper2DATA\inputdata\landuse\2012_to2020.tif",
        r"D:\paper2DATA\inputdata\landuse\2016_to2020.tif",
        
        # 标签数据
        r"D:\paper2DATA\inputdata\convlstm\12label\12label_to2020.tif",
        r"D:\paper2DATA\inputdata\convlstm\16label\16label_to2020.tif",
        r"D:\paper2DATA\inputdata\convlstm\20label\20label_to2020.tif",
        
        # 因子数据
        r"D:\paper2DATA\inputdata\convlstm\floodrisk\floodrisk_to2020.tif",
        r"D:\paper2DATA\inputdata\convlstm\greenspace\greenspace_to2020.tif",
        r"D:\paper2DATA\inputdata\convlstm\fibre\fibre_to2020.tif",
        r"D:\paper2DATA\inputdata\convlstm\slope_to2020.tif",
        r"D:\paper2DATA\inputdata\convlstm\3water1000m_to2020.tif",
        r"D:\paper2DATA\inputdata\convlstm\dem\dem_to2020.tif",
        r"D:\paper2DATA\inputdata\busstop\busstop_to2020.tif",
        r"D:\paper2DATA\inputdata\roads\roads_to2020.tif",
    ]
    
    # 运行检查
    check_raster_consistency(reference, files_to_check)