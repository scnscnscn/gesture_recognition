import numpy as np
import joblib

# 加载训练好的 SVM 模型
def load_model(model_path):
    return joblib.load(model_path)

# 使用傅里叶描述子进行分类
def test_fd(fd_test, model_path):
    # 加载模型
    model = load_model(model_path)
    # 预测
    prediction = model.predict(fd_test)
    return prediction

# 使用椭圆傅里叶描述子进行分类
def test_efd(efd_test, model_path):
    # 加载模型
    model = load_model(model_path)
    # 预测
    prediction = model.predict(efd_test)
    return prediction

