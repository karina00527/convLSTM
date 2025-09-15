import rasterio
import numpy as np
import matplotlib.pyplot as plt

fp = "datas/coastline/coastline.tif"
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
