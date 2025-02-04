import random
import cv2
import os

# 旋转函数
def rotate(image, scale=0.9):
    angle = random.randrange(-90, 90)  # 随机角度
    w = image.shape[1]
    h = image.shape[0]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    image = cv2.warpAffine(image, M, (w, h))
    return image

def process_images_in_directory(base_path, category):
    """
    处理指定类别目录下的图像，生成旋转和翻转后的图像。
    :param base_path: 基础路径
    :param category: 类别目录名
    """
    category_path = os.path.join(base_path, category)
    if not os.path.exists(category_path):
        print(f"目录 {category_path} 不存在，跳过")
        return

    cnt = 21  # 计数器，从21开始，避免与原始图像冲突
    for j in range(1, 21):  # 每个类别有20张原始图像
        img_path = os.path.join(category_path, f"{category}_{j}.png")
        roi = cv2.imread(img_path)
        if roi is None:
            print(f"图像 {img_path} 无法读取，跳过")
            continue

        for k in range(12):  # 对每张图像生成12次旋转和翻转
            img_rotation = rotate(roi)  # 旋转
            new_rotation_path = os.path.join(category_path, f"{category}_{cnt}.png")
            cv2.imwrite(new_rotation_path, img_rotation)
            cnt += 1

            img_flip = cv2.flip(img_rotation, 1)  # 水平翻转
            new_flip_path = os.path.join(category_path, f"{category}_{cnt}.png")
            cv2.imwrite(new_flip_path, img_flip)
            cnt += 1

        print(f"已完成 {category}_{j}")

if __name__ == "__main__":
    base_path = './image'  # 基础路径
    categories = [str(i) for i in range(1, 10)]  # 处理类别1到9

    for category in categories:
        process_images_in_directory(base_path, category)