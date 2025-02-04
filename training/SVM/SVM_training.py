import os
import numpy as np
from os import listdir
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# 全局路径
path = './feature/'
model_path = "./model/"
test_path = "./test_feature/"

# 读取txt文件并将每个文件的描述子改为一维的矩阵存储
def txtToVector(filename, N):
    returnVec = np.zeros((1, N))
    fr = open(filename)
    lineStr = fr.readline()
    lineStr = lineStr.split(' ')
    for i in range(N):
        returnVec[0, i] = int(lineStr[i])
    return returnVec

# 训练SVM模型
def train_SVM(N):
    # 检查训练路径是否存在，如果不存在则创建
    if not os.path.exists(path):
        print(f"训练路径不存在：{path}，正在创建...")
        os.makedirs(path, exist_ok=True)
        print(f"路径已创建：{path}")

    # 加载训练数据
    hwLabels = []  # 存放类别标签
    trainingFileList = listdir(path)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, N))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i, :] = txtToVector(path + fileNameStr, N)  # 将训练集改为矩阵格式
    print("数据加载完成")

    # 定义SVM模型参数
    svc = SVC()
    parameters = {
        'kernel': ('linear', 'rbf'),
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]
    }

    # 使用网格搜索法进行参数优化
    clf = GridSearchCV(svc, parameters, cv=5, n_jobs=8)  # 设置5折交叉验证
    clf.fit(trainingMat, hwLabels)
    print("最佳参数:", clf.best_params_)  # 打印出最好的结果

    # 保存最佳模型
    save_path = model_path + "svm_efd_" + "train_model.m"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(clf.best_estimator_, save_path)
    print("SVM Model saved successfully.")

# 测试SVM模型
def test_SVM(N):
    # 检查测试路径是否存在，如果不存在则创建
    if not os.path.exists(test_path):
        print(f"测试路径不存在：{test_path}，正在创建...")
        os.makedirs(test_path, exist_ok=True)
        print(f"路径已创建：{test_path}")

    # 加载模型
    clf = joblib.load(model_path + "svm_efd_" + "train_model.m")

    # 加载测试数据
    testFileList = listdir(test_path)
    if not testFileList:
        print("测试路径下没有文件，请检查路径或添加测试文件。")
        return

    errorCount = 0  # 记录错误个数
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNum = int(fileNameStr.split('_')[0])
        vectorTest = txtToVector(test_path + fileNameStr, N)
        valTest = clf.predict(vectorTest)
        if valTest != classNum:
            errorCount += 1
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))

if __name__ == "__main__":
    # 假设特征维度为31
    feature_dim = 31
    train_SVM(feature_dim)  # 训练模型
    test_SVM(feature_dim)   # 测试模型