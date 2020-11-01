
# script will train a digit OCR model on the MNIST dataset.
# 训练 MNIST OCR 模型

# python train_digit_classifier.py --model output/digit_classifier.h5

from pyimagesearch.models import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model after training")
args = vars(ap.parse_args())

INIT_LR = 1e-3
EPOCHS = 10
BS = 128

print("[INFO] accessing MNIST...")
((trainData, trainLabels),(testData, testLabels)) = mnist.load_data()

# print(trainData.shape,trainData[:1])
trainData = trainData.reshape((trainData.shape[0],28,28,1)) # 转换为[indax, x, y, channel] 的格式
testData = testData.reshape((testData.shape[0],28,28,1))
# print(trainData.shape,trainData[:1])

# scale data to the range of [0,1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# convert the labels from integers to vectors
le = LabelBinarizer()
tls = trainLabels
trainLabels = le.fit_transform(trainLabels) # 注册标签枚举并转换为二进制表
testLabels = le.transform(testLabels)

print("tls trainLabels: ",tls , trainLabels[0])


print("[INFO] compiling model...")
opt = Adam(lr = INIT_LR) # https://keras.io/zh/optimizers/#adam
model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", # 编译模型 # 损失函数用分类交叉熵
    optimizer=opt, # 优化器
    metrics=["accuracy"])#训练结束后显示的评估数据

# train the network
print("[INFO] training network...")
H = model.fit( # 训练模型
	trainData, trainLabels,
	validation_data=(testData, testLabels), # 验证数据
	batch_size=BS, # 计算多少个样本的误差做一次模型更新
	epochs=EPOCHS, # 超参数 所有数据训练几轮
	verbose=1) # 进度条模式

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testData) # 应用模型 预测数据
print("argmax: ",
	testLabels.argmax(axis=1),
	predictions.argmax(axis=1),)
print(classification_report(
	testLabels.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))
# serialize the model to disk
print("[INFO] serializing digit model...")
model.save(args["model"], save_format="h5") # 模型保存为参数指定的文件

print('**END**')

