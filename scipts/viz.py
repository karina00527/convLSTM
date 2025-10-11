import rasterio
import numpy as np
import matplotlib.pyplot as plt

    # "/home/xyf/Downloads/landuse/datas/mask/mask.tif",
    # "ANN/2008urban.tif",  # 1 = Urban, 0 = NonUrban (按你 mapping)
    # "/home/xyf/Downloads/landuse/datas/greenspace/greenspace_to2020.tif",
    # "/home/xyf/Downloads/landuse/datas/fibre/fibre_to2020.tif",
    # "/home/xyf/Downloads/landuse/datas/3water1000m.tif",
    # "/home/xyf/Downloads/landuse/datas/floodrisk/floodrisk_to2020.tif",
    # "/home/xyf/Downloads/landuse/datas/roads/roads_to2020.tif",
    # "ANN/income13.tif",
    # "ANN/income18.tif",
    # "/home/xyf/Downloads/landuse/datas/sitesize.tif",
    # "/home/xyf/Downloads/landuse/datas/busstop_to2020.tif",
    # "/home/xyf/Downloads/landuse/datas/school_to2020.tif",
    # "/home/xyf/Downloads/landuse/datas/hospital_to2020.tif",
    # "/home/xyf/Downloads/landuse/datas/shoppingmall/shoppingmal_to2020.tif",

fp = "/home/xyf/Downloads/landuse/convLSTM/scipts/agent_based_model/output/combined_three_values.tif"
# fp = "/home/xyf/Downloads/landuse/datas/greenspace/greenspace.tif"
with rasterio.open(fp) as ds:
    arr = ds.read(1).astype(float)
    nodata = ds.nodata
    if nodata is not None:
        arr = np.where(arr==nodata, np.nan, arr)
    print("CRS:", ds.crs)
    print("Pixel size:", ds.transform.a, ds.transform.e)
    print("Min/Max (ignoring NaN):", np.nanmin(arr), np.nanmax(arr))

plt.figure()
im = plt.imshow(arr, cmap='viridis')   # 伪彩显示
plt.colorbar(im, label='Distance to nearest greenspace')
plt.title('greenspace.tif')
plt.axis('off')
plt.show()

# 直方图，了解分布（可判断是否要对数/标准化）
finite = np.isfinite(arr)
plt.figure()
plt.hist(arr[finite], bins=50)
plt.xlabel('Distance'); plt.ylabel('Count'); plt.title('Histogram')
plt.show()
