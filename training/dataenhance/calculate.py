import os
import cv2
from gesture_record import image_processor as ip
import numpy as np

path = 'dataenhance/feature/'
path_img = './image/'

if __name__ == "__main__":
    # 确保特征目录存在
    if not os.path.exists(path):
        os.makedirs(path)

    # 获取所有子文件夹（类别）
    categories = os.listdir(path_img)
    categories = [cat for cat in categories if os.path.isdir(os.path.join(path_img, cat))]

    for cat in categories:  # 遍历每个类别
        cat_path = os.path.join(path_img, cat)
        for img_name in os.listdir(cat_path):  # 遍历类别文件夹中的所有图像
            img_path = os.path.join(cat_path, img_name)
            roi = cv2.imread(img_path)
            if roi is None:
                print(f"图像 {img_path} 无法读取，跳过")
                continue

            # 将图像转换为灰度图像
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 调用 fourierDescriptor 函数
            result = ip.fourierDescriptor(roi_gray)
            print("fourierDescriptor 返回值:", result)  # 调试输出

            # 提取傅里叶描述子（假设是返回值的第二个元素）
            if isinstance(result, tuple) and len(result) > 1:
                descriptor_in_use = result[1]  # 傅里叶描述子
            else:
                print(f"图像 {img_path} 的傅里叶描述子无效，跳过")
                continue

            # 检查描述子是否有效
            if len(descriptor_in_use) < 2:
                print(f"图像 {img_path} 的傅里叶描述子无效，跳过")
                continue

            # 构造特征文件名
            fd_name = os.path.join(path, f"{cat}_{os.path.splitext(img_name)[0]}.txt")
            with open(fd_name, 'w', encoding='utf-8') as f:
                # 取傅里叶描述子的绝对值（模）
                descriptor_magnitude = np.abs(descriptor_in_use)

                # 检查归一化基准值是否为零
                temp = descriptor_magnitude[1]
                if temp == 0:
                    print(f"图像 {img_path} 的归一化基准值为零，跳过")
                    continue

                for k in range(1, len(descriptor_magnitude)):
                    x_record = int(100 * descriptor_magnitude[k] / temp)
                    f.write(str(x_record) + ' ')
                f.write('\n')

            print(f"已完成 {cat}_{img_name}")

        print(f"类别 {cat} 的所有图像处理完成")