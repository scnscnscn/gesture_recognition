import cv2
import numpy as np
import os

MIN_DESCRIPTOR = 32  # 保留的傅里叶描述子数量

def preprocess_image(img):
    """图像预处理：肤色检测和形态学操作。"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    cr = cv2.split(ycrcb)[1]
    cr_blurred = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_eroded = cv2.erode(skin, kernel, iterations=1)
    skin_dilated = cv2.dilate(skin_eroded, kernel, iterations=2)
    return skin_dilated

def detect_contours(img, skin_dilated):
    """提取并绘制轮廓。"""
    contours = find_contours(skin_dilated)
    return cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)

def fourierDescriptor(skin_dilated):
    """计算傅里叶描述子"""
    contours = find_contours(skin_dilated)
    if not contours:
        return None, None

    # 取面积最大的轮廓
    contour = max(contours, key=cv2.contourArea)
    contour_array = contour[:, 0, :]

    # 创建轮廓图像
    ret_np = np.zeros(skin_dilated.shape, np.uint8)
    ret = cv2.drawContours(ret_np, [contour], -1, 255, 1)

    # 转换为复数形式
    contours_complex = contour_array[:, 0] + 1j * contour_array[:, 1]

    # 傅里叶变换并截断
    fourier_result = np.fft.fft(contours_complex)
    descriptor = truncate_descriptor(fourier_result)

    return ret, descriptor

def find_contours(binary_img):
    """处理不同OpenCV版本的findContours差异"""
    cnts = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    return sorted(contours, key=cv2.contourArea, reverse=True)

def truncate_descriptor(fourier_result):
    """截断并保留低频成分"""
    num_coeffs = len(fourier_result)
    center = num_coeffs // 2
    low, high = center - MIN_DESCRIPTOR//2, center + MIN_DESCRIPTOR//2
    descriptor = np.fft.fftshift(fourier_result)
    descriptor = descriptor[low:high]
    return np.fft.ifftshift(descriptor)

def binaryMask(frame, x0, y0, width, height):
    """处理摄像头捕获的图像并返回结果"""
    # 提取 ROI 区域
    roi = frame[y0:y0+height, x0:x0+width]
    if roi.size == 0:
        return None, None, None, None, None

    # 预处理图像
    skin_dilated = preprocess_image(roi)

    # 检测轮廓
    contour_img = detect_contours(roi, skin_dilated)

    # 计算傅里叶描述子
    ret, fourier_result = fourierDescriptor(skin_dilated)

    # 返回结果
    return roi, contour_img, ret, fourier_result, None

def reconstruct(img, descriptor):
    """从傅里叶描述子重建轮廓"""
    coeffs = np.fft.ifftshift(descriptor)
    contour_reconstruct = np.fft.ifft(coeffs)

    # 转换为坐标点
    contour_points = np.array([
        [np.real(pt), np.imag(pt)]
        for pt in contour_reconstruct
    ], dtype=np.float32)

    # 归一化到图像尺寸
    contour_points -= np.min(contour_points, axis=0)
    scale = min(img.shape[0]/np.max(contour_points[:,1]),
                img.shape[1]/np.max(contour_points[:,0]))
    contour_points = (contour_points * scale).astype(np.int32)

    # 绘制轮廓
    black_np = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.polylines(black_np, [contour_points], isClosed=True, color=255, thickness=1)
    return black_np


def process_image(image_path, output_folder=None, folder_num=None, image_counter=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None

    skin_dilated = preprocess_image(img)
    contour_img = detect_contours(img, skin_dilated)
    ret, descriptor = fourierDescriptor(skin_dilated)

    if descriptor is not None:
        reconstructed = reconstruct(img, descriptor)

    # 保存和显示结果
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        base = f"{folder_num}-{image_counter}"  # 使用文件夹编号和图像计数器生成文件名
        if ret is not None:
            cv2.imwrite(f"{output_folder}/{base}_fourier.png", ret)


    # 显示结果
    def resize(img, width=800):
        h, w = img.shape[:2]
        if w > width:
            return cv2.resize(img, (width, int(h * width / w)))
        return img

    cv2.imshow("Original", resize(img))
    cv2.imshow("Skin", resize(skin_dilated))
    cv2.imshow("Contour", resize(contour_img))
    if ret is not None:
        cv2.imshow("Fourier", resize(ret))
    if 'reconstructed' in locals():
        cv2.imshow("Reconstructed", resize(reconstructed))

    key = cv2.waitKey(20)
    cv2.destroyAllWindows()
    return key


def process_images_in_folder(folder_path, output_folder=None):
    if not os.path.isdir(folder_path):
        print(f"路径不存在: {folder_path}")
        return

    # 提取文件夹路径的结尾数字
    folder_num = os.path.basename(os.path.normpath(folder_path))
    if not folder_num.isdigit():
        print(f"文件夹路径的结尾不是数字: {folder_num}")
        return

    # 初始化图像计数器
    image_counter = 1

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            if os.path.isfile(image_path):
                print(f"处理中: {filename}")
                key = process_image(image_path, output_folder, folder_num, image_counter)
                image_counter += 1  # 更新图像计数器
                if key == 27:
                    print("用户终止处理")
                    return


if __name__ == "__main__":
    input_folder = input("输入图像文件夹路径: ")
    output_folder = os.path.join(input_folder, "processed_results")
    process_images_in_folder(input_folder, output_folder)