import time
import cv2
import os
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class ROIProcessor:
    '''ROI处理器'''
    def __init__(self, roi_size=(300, 300)):
        self.roi_width, self.roi_height = roi_size
        self.x0, self.y0 = 300, 100

    def get_roi(self, frame):
        """获取ROI区域"""
        return frame[self.y0:self.y0 + self.roi_height, self.x0:self.x0 + self.roi_width]

    def process_roi(self, roi):
        """处理ROI区域并返回结果"""
        return roi

    def move_roi(self, frame, direction):
        """移动ROI区域"""
        if direction == 'up':
            self.y0 = max(0, self.y0 - 10)
        elif direction == 'down':
            self.y0 = min(frame.shape[0] - self.roi_height, self.y0 + 10)
        elif direction == 'left':
            self.x0 = max(0, self.x0 - 10)
        elif direction == 'right':
            self.x0 = min(frame.shape[1] - self.roi_width, self.x0 + 10)


class UIManager:
    def __init__(self, roi_processor):
        self.roi_processor = roi_processor
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_size = 0.5
        self.fx, self.fy = 10, 355
        self.fh = 18
        self.roi_color = (0, 255, 0)
        self.text_color = (255, 255, 255)

    @staticmethod
    def add_chinese_text(img, text, position, textColor=(255, 255, 255), textSize=30):
        """ 添加中文文字 """
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
            except IOError:
                fontStyle = ImageFont.truetype("arial.ttf", textSize)
            draw.text(position, text, fill=textColor, font=fontStyle)
            return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logging.error(f"文字渲染失败: {str(e)}")
            return img

    def update_ui(self, frame, gesture_name, counter, num_samples, recording):
        """更新用户界面"""
        frame = self.add_chinese_text(frame, "操作指引:", (self.fx, self.fy), self.text_color, 20)
        y_pos = self.fy + 30
        tips = [
            "I/J/K/L - 移动ROI区域",
            "N - 新建手势类别",
            "S - 开始/停止录制",
            "Q - 退出系统",
            f"当前手势: {gesture_name if gesture_name else '未设置'}",
            f"采集进度: [{counter}/{num_samples}]" if recording else "等待录制..."
        ]

        for tip in tips:
            frame = self.add_chinese_text(frame, tip, (self.fx, y_pos), self.text_color, 18)
            y_pos += 25

        # 画出ROI区域的黑框
        cv2.rectangle(frame, (self.roi_processor.x0, self.roi_processor.y0),
                      (self.roi_processor.x0 + self.roi_processor.roi_width,
                       self.roi_processor.y0 + self.roi_processor.roi_height),
                      (0, 0, 0), 2)

        return frame

class GestureCollector:
    def __init__(self, camera_index=0, num_samples=20, roi_size=(300, 300)):
        self.camera_index = camera_index
        self.num_samples = num_samples
        self.counter = 0
        self.gesture_name = ''
        self.save_path = ''
        self.binary_mode = False
        self.recording = False
        self.cap = None
        self.roi_processor = ROIProcessor(roi_size)
        self.ui_manager = UIManager(self.roi_processor)
        self.setup_logging()

    def setup_logging(self):
        """ 设置日志记录 """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

    def start(self):
        """ 开始采集手势 """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logging.error("无法打开摄像头")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("视频采集失败")
                break

            frame = self.process_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()

    def process_frame(self, frame):
        """ 处理视频帧 """
        frame = cv2.flip(frame, 1)
        roi = self.roi_processor.get_roi(frame)
        display_roi = self.roi_processor.process_roi(roi)

        frame = self.ui_manager.update_ui(frame, self.gesture_name, self.counter, self.num_samples, self.recording)
        cv2.imshow("Gesture Collection", frame)
        cv2.imshow("ROI Preview", display_roi)

        self.handle_key_events(frame)
        self.save_samples(roi)

        return frame

    def handle_key_events(self, frame):
        """ 处理按键事件 """
        key = cv2.waitKey(1) & 0xFF
        if key == ord('i'):
            self.roi_processor.move_roi(frame, 'up')
        elif key == ord('k'):
            self.roi_processor.move_roi(frame, 'down')
        elif key == ord('j'):
            self.roi_processor.move_roi(frame, 'left')
        elif key == ord('l'):
            self.roi_processor.move_roi(frame, 'right')
        elif key == ord('n'):
            self.create_gesture_folder()
        elif key == ord('s'):
            self.toggle_recording()

    def ask_for_next_gesture(self):
        """询问用户是否继续创建新的手势类别"""
        response = input("是否继续创建新的手势类别？(y/n): ").strip().lower()
        if response == 'y':
            self.counter = 0  # 重置计数器
            self.recording = False  # 先停止录制
            self.create_gesture_folder()  # 创建新手势类别
        elif response == 'n':
            logging.info("程序正常退出")
            self.release_resources()
            exit(0)
        else:
            logging.warning("输入无效，请输入 'y' 或 'n'。")
            self.ask_for_next_gesture()

    def create_gesture_folder(self):
        """创建新手势文件夹"""
        self.gesture_name = input("请输入新手势名称（控制台输入）: ")
        self.save_path = os.path.join("data", self.gesture_name)
        os.makedirs(self.save_path, exist_ok=True)
        logging.info(f"新建手势类别: {self.gesture_name}")
        self.counter = 0  # 确保计数器重置
        self.recording = True  # 开始录制

    def toggle_recording(self):
        """ 开始/停止录制 """
        if not self.gesture_name:
            logging.warning("请先创建手势类别")
            return
        self.recording = not self.recording
        if self.recording:
            self.counter = 0  # 重置计数器
        logging.info(f"{'开始' if self.recording else '停止'}录制")

    def save_samples(self, roi):
        """ 保存样本 """
        if self.recording and self.counter < self.num_samples:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)

            save_path = os.path.join(self.save_path, f"{self.gesture_name}_{self.counter:04d}.png")
            cv2.imwrite(save_path, roi)
            self.counter += 1
            logging.info(f"已保存样本 {self.counter}/{self.num_samples}")
            time.sleep(0.05)

        elif self.counter >= self.num_samples:
            self.recording = False
            logging.info(f"已完成{self.gesture_name}的{self.num_samples}张采集")
            self.ask_for_next_gesture()

    def release_resources(self):
        """ 释放资源 """
        self.cap.release()
        cv2.destroyAllWindows()
        logging.info("程序正常退出")


if __name__ == "__main__":
    collector = GestureCollector()
    collector.start()