import cv2
import os
import numpy as np
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化参数
font = cv2.FONT_HERSHEY_SIMPLEX  # 字体样式
size = 0.5  # 字体大小
fx = 10  # 提示信息的x坐标起始位置
fy = 355  # 提示信息的y坐标起始位置
fh = 18  # 提示信息的行高
x0 = 300  # ROI区域的x坐标起始位置
y0 = 100  # ROI区域的y坐标起始位置
width = 300  # ROI区域的宽度
height = 300  # ROI区域的高度
numofsamples = 300  # 每个手势需要录制的样本数
counter = 0  # 已录制的图片计数器
gesturename = ''  # 手势名称，用于创建文件夹
path = ''  # 保存手势图片的路径
binaryMode = False  # 是否应用肤色检测
saveImg = False  # 是否保存图片

# 创建视频捕捉对象
cap = cv2.VideoCapture(0)  # 使用内置摄像头

def save_roi(img):
    """
    保存ROI区域的图像。
    """
    global path, counter, gesturename, saveImg
    if counter >= numofsamples:
        saveImg = False
        gesturename = ''
        counter = 0
        return

    counter += 1
    name = gesturename + str(counter)  # 给录制的手势命名
    logging.info(f"Saving img: {name}")
    cv2.imwrite(os.path.join(path, name + '.png'), img)  # 写入文件
    time.sleep(0.05)  # 等待50毫秒

def draw_text(frame):
    """
    在视频帧上绘制提示信息。
    """
    cv2.putText(frame, "Option: ", (fx, fy), font, size, (0, 255, 0))
    cv2.putText(frame, "i/j - Move ROI", (fx, fy + fh), font, size, (0, 255, 0))
    cv2.putText(frame, "k/l - Resize ROI", (fx, fy + 2 * fh), font, size, (0, 255, 0))
    cv2.putText(frame, "s - Save ROI", (fx, fy + 4 * fh), font, size, (0, 255, 0))
    cv2.putText(frame, "q - Quit", (fx, fy + 5 * fh), font, size, (0, 255, 0))

def main():
    global x0, y0, width, height, binaryMode, saveImg, gesturename, path

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame")
            break

        frame = cv2.flip(frame, 2)  # 图像翻转
        roi = frame[y0:y0 + height, x0:x0 + width]  # 提取ROI区域

        draw_text(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if gesturename:
                saveImg = True
            else:
                logging.warning("Enter a gesture group name first, by pressing 'n'!")
                saveImg = False
        elif key == ord('n'):
            gesturename = input("Enter the gesture folder name: ")
            os.makedirs(gesturename, exist_ok=True)
            path = os.path.join(".", gesturename)
        elif key == ord('q'):
            break
        elif key == ord('i'):
            y0 = max(0, y0 - 10)  # 确保y0不小于0
        elif key == ord('k'):
            y0 = min(frame.shape[0] - height, y0 + 10)  # 确保y0+height不大于frame的高度
        elif key == ord('j'):
            x0 = max(0, x0 - 10)  # 确保x0不小于0
        elif key == ord('l'):
            x0 = min(frame.shape[1] - width, x0 + 10)  # 确保x0+width不大于frame的宽度
        elif key == ord('m'):
            binaryMode = not binaryMode

        cv2.imshow('frame', frame)
        cv2.imshow("ROI", roi)

        if saveImg:
            save_roi(roi)

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Program exited")

if __name__ == "__main__":
    main()