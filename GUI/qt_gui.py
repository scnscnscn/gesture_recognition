from gesture_record import image_processor as ip
from PyQt5.QtWidgets import QApplication, QWidget, QToolTip, QPushButton, QMessageBox, QDesktopWidget, QLabel
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2
import classify as cf
import numpy as np
import sys

class MyWindow(QWidget):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.timer_camera = QTimer()
        self.cap = cv2.VideoCapture()
        self.initUI()
        self.slot_init()

    def initUI(self):
        self.mylabel()
        self.myButton()
        self.myLabelPic()
        self.setFixedSize(670, 520)
        self.center()
        self.setWindowIcon(QIcon('icon.jpg'))
        self.setWindowTitle('Gesture Recognition')

    def mylabel(self):
        label_roi = QLabel('原图', self)
        label_roi.setStyleSheet("QLabel{font-size:18px;}")
        label_roi.resize(60, 30)
        label_roi.move(120, 15)

        label_res = QLabel('轮廓线', self)
        label_res.setStyleSheet("QLabel{font-size:18px;}")
        label_res.resize(60, 30)
        label_res.move(480, 15)

        label_pre = QLabel('预测', self)
        label_pre.setStyleSheet("QLabel{font-size:20px;}")
        label_pre.resize(50, 30)
        label_pre.move(400, 400)

        label_result = QLabel('结果', self)
        label_result.setStyleSheet("QLabel{font-size:20px;}")
        label_result.resize(50, 30)
        label_result.move(400, 430)

    def myLabelPic(self):
        self.label_show_roi = QLabel(self)
        self.label_show_roi.setFixedSize(301, 301)
        self.label_show_roi.move(20, 50)
        self.label_show_roi.setStyleSheet("QLabel{background:white;}")
        self.label_show_roi.setAutoFillBackground(True)

        self.label_show_ret = QLabel(self)
        self.label_show_ret.setFixedSize(301, 301)
        self.label_show_ret.move(350, 50)
        self.label_show_ret.setStyleSheet("QLabel{background:white;}")
        self.label_show_ret.setAutoFillBackground(True)

        self.label_show_recognition = QLabel('0', self)
        self.label_show_recognition.setStyleSheet("QLabel{background:white; font-size:50px;}")
        self.label_show_recognition.setFixedSize(100, 100)
        self.label_show_recognition.move(500, 380)
        self.label_show_recognition.setAutoFillBackground(True)

    def myButton(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        self.button_open_camera = QPushButton('打开相机', self)
        self.button_open_camera.setToolTip('按i,k,j,l可以进行上下左右调整')
        self.button_open_camera.resize(100, 30)
        self.button_open_camera.move(100, 400)

        self.button_recognition = QPushButton('开始预测', self)
        self.button_recognition.setFixedSize(100, 30)
        self.button_recognition.move(100, 450)

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.button_recognition.clicked.connect(self.button_recognition_click)
        self.timer_camera.timeout.connect(self.show_camera)

    def button_open_camera_click(self):
        try:
            if not self.timer_camera.isActive():
                self.cap.open(0)
                self.timer_camera.start(30)
                self.button_open_camera.setText('关闭相机')
            else:
                self.timer_camera.stop()
                self.cap.release()
                self.label_show_roi.clear()
                self.label_show_ret.clear()
                self.label_show_recognition.setText('0')
                self.button_open_camera.setText('打开相机')
        except Exception as e:
            QMessageBox.warning(self, "错误", f"打开或关闭摄像头时出错：{str(e)}")

    def button_recognition_click(self):
        try:
            if not hasattr(self, 'fourier_result') or self.fourier_result is None:
                QMessageBox.warning(self, "警告", "请先打开摄像头并采集图像。")
                return

            descriptor_in_use = abs(self.fourier_result)
            fd_test = np.zeros((1, 31))
            temp = descriptor_in_use[1]
            for k in range(1, len(descriptor_in_use)):
                fd_test[0, k - 1] = int(100 * descriptor_in_use[k] / temp)

            model_path = "C:/Users/WLQVi/Desktop/python/gesture_recognition/GUI/model/svm_efd_train_model.m"
            prediction = cf.test_efd(fd_test, model_path)  # 获取模型预测结果

            print(f"Model prediction: {prediction}")  # 打印模型预测结果

            num = [0] * 11
            num[prediction[0]] += 1  # 根据预测结果更新计数

            res = 0
            for i in range(1, 11):
                if num[i] >= 1:
                    res = i
                    break

            self.label_show_recognition.setText(str(res))
        except Exception as e:
            QMessageBox.warning(self, "错误", f"手势识别时出错：{str(e)}")

    def show_camera(self):
        try:
            width, height = 300, 300  # 设置拍摄窗口大小
            x0, y0 = 300, 100  # 设置选取位置
            flag, frame = self.cap.read()
            if not flag:
                QMessageBox.warning(self, "警告", "无法读取摄像头图像。")
                self.timer_camera.stop()
                return

            roi, res, ret, self.fourier_result, _ = ip.binaryMask(frame, x0, y0, width, height)
            if roi is None:
                return

            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            show_roi = QImage(roi.data, roi.shape[1], roi.shape[0], QImage.Format_RGB888)
            show_ret = QImage(ret.data, ret.shape[1], ret.shape[0], QImage.Format_Grayscale8)
            self.label_show_roi.setPixmap(QPixmap.fromImage(show_roi))
            self.label_show_ret.setPixmap(QPixmap.fromImage(show_ret))
        except Exception as e:
            QMessageBox.warning(self, "错误", f"显示摄像头图像时出错：{str(e)}")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
        else:
            event.ignore()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec())