import csv
import numpy as np
from PIL import Image

def readTrafficSigns(rootpath):
    """读取交通标志数据集，返回形状为（样本数,32,32,3）的标准化数组"""
    images = []
    labels = []

    for c in range(43):
        prefix = f"{rootpath}/{c:05d}/"
        with open(f"{prefix}GT-{c:05d}.csv", "r") as gtFile:
            gtReader = csv.reader(gtFile, delimiter=';')
            next(gtReader)  # 跳过表头

            for row in gtReader:
                # 1. 加载图像并强制尺寸为32x32
                img = Image.open(prefix + row[0]).resize((32, 32))
                img_array = np.array(img, dtype=np.uint8)
                
                # 2. 通道标准化处理
                if img_array.ndim == 2:  # 灰度图转RGB
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 4:  # 去除Alpha通道
                    img_array = img_array[:, :, :3]
                
                # 3. 添加维度验证断言
                assert img_array.shape == (32,32,3), f"图像{row[0]}尺寸异常: {img_array.shape}"
                
                images.append(img_array)
                labels.append(int(row[7]))

    # 4. 转换时显式指定轴顺序
    images_np = np.transpose(np.array(images), (0,1,2,3))  # 维度顺序修正[1,5](@ref)
    labels_np = np.array(labels, dtype=np.int32)
    return images_np, labels_np