import os
import cv2
import numpy as np
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 图像预处理类
class ImageProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, img):
        """图像预处理：肤色检测和形态学操作。"""
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        cr = cv2.split(ycrcb)[1]
        cr_blurred = cv2.GaussianBlur(cr, (5, 5), 0)
        _, skin = cv2.threshold(cr_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_eroded = cv2.erode(skin, kernel, iterations=1)
        skin_dilated = cv2.dilate(skin_eroded, kernel, iterations=2)
        self.logger.info("图像预处理完成")
        return skin_dilated

    def detect_contours(self, img, skin_dilated):
        """提取并绘制轮廓。"""
        contours = self.find_contours(skin_dilated)
        contour_img = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
        self.logger.info("轮廓提取完成")
        return contour_img

    def find_contours(self, binary_img):
        """处理不同OpenCV版本的findContours差异"""
        cnts = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]
        return sorted(contours, key=cv2.contourArea, reverse=True)

# 傅里叶描述子类
class FourierDescriptor:
    MIN_DESCRIPTOR = 32  # 保留的傅里叶描述子数量

    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.logger = logging.getLogger(__name__)

    def calculate(self, skin_dilated):
        """计算傅里叶描述子"""
        contours = self.image_processor.find_contours(skin_dilated)
        if not contours:
            self.logger.warning("未检测到轮廓")
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
        descriptor = self.truncate_descriptor(fourier_result)

        self.logger.info("傅里叶描述子计算完成")
        return ret, descriptor

    def truncate_descriptor(self, fourier_result):
        """截断并保留低频成分"""
        num_coeffs = len(fourier_result)
        center = num_coeffs // 2
        low, high = center - self.MIN_DESCRIPTOR // 2, center + self.MIN_DESCRIPTOR // 2
        descriptor = np.fft.fftshift(fourier_result)
        descriptor = descriptor[low:high]
        return np.fft.ifftshift(descriptor)

    def reconstruct(self, img, descriptor):
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
        scale = min(img.shape[0] / np.max(contour_points[:, 1]),
                    img.shape[1] / np.max(contour_points[:, 0]))
        contour_points = (contour_points * scale).astype(np.int32)

        # 绘制轮廓
        black_np = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.polylines(black_np, [contour_points], isClosed=True, color=255, thickness=1)
        self.logger.info("轮廓重建完成")
        return black_np

# 手势处理类
class GestureProcessor:
    def __init__(self, input_folder="data", output_folder="dataset"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.image_processor = ImageProcessor()
        self.fourier_descriptor = FourierDescriptor(self.image_processor)
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.output_folder, exist_ok=True)

    def process_image(self, image_path, output_folder, folder_num, image_counter):
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"无法读取图像: {image_path}")
            return

        skin_dilated = self.image_processor.preprocess_image(img)
        contour_img = self.image_processor.detect_contours(img, skin_dilated)
        ret, descriptor = self.fourier_descriptor.calculate(skin_dilated)

        if descriptor is not None:
            reconstructed = self.fourier_descriptor.reconstruct(img, descriptor)

        # 保存结果
        os.makedirs(output_folder, exist_ok=True)
        base = f"{folder_num}-{image_counter}"
        if ret is not None:
            cv2.imwrite(f"{output_folder}/{base}.png", ret)

        self.logger.info(f"处理完成: {image_path}")

    def process_images_in_folder(self, folder_path, output_folder):
        folder_num = os.path.basename(os.path.normpath(folder_path))
        if not folder_num.isdigit():
            self.logger.warning(f"文件夹路径的结尾不是数字: {folder_num}")
            return

        image_counter = 1
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(folder_path, filename)
                if os.path.isfile(image_path):
                    self.process_image(image_path, output_folder, folder_num, image_counter)
                    image_counter += 1

    def run(self):
        for folder_name in os.listdir(self.input_folder):
            folder_path = os.path.join(self.input_folder, folder_name)
            if os.path.isdir(folder_path):
                output_path = os.path.join(self.output_folder, folder_name)
                self.process_images_in_folder(folder_path, output_path)

# 主程序入口
if __name__ == "__main__":
    setup_logging()
    processor = GestureProcessor()
    processor.run()