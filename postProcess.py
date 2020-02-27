import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from dataloader import DataLoader
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


class TrainProcess:
	def __init__(self, path):
		self.trainLoss = np.array([])
		self.trainAcc = np.array([])
		self.testLoss = np.array([])
		self.testAcc = np.array([])
		self.dataDir = os.path.join(path, "data")
		self.imgDir = os.path.join(path, "img")

	def addData(self, trainLoss, trainAcc, testLoss, testAcc):
		self.trainLoss = np.append(self.trainLoss, trainLoss)
		self.trainAcc = np.append(self.trainAcc, trainAcc)
		self.testLoss = np.append(self.testLoss, testLoss)
		self.testAcc = np.append(self.testAcc, testAcc)

	def save(self):
		f = open(os.path.join(self.dataDir, "trainLoss.pkl"), "wb")
		pickle.dump(self.trainLoss, f)
		f = open(os.path.join(self.dataDir, "trainAcc.pkl"), "wb")
		pickle.dump(self.trainAcc, f)
		f = open(os.path.join(self.dataDir, "testLoss.pkl"), "wb")
		pickle.dump(self.testLoss, f)
		f = open(os.path.join(self.dataDir, "testAcc.pkl"), "wb")
		pickle.dump(self.testAcc, f)

	def restore(self):
		f = open(os.path.join(self.dataDir, "trainLoss.pkl"), "rb")
		self.trainLoss = pickle.load(f)
		f = open(os.path.join(self.dataDir, "trainAcc.pkl"), "rb")
		self.trainAcc = pickle.load(f)
		f = open(os.path.join(self.dataDir, "testLoss.pkl"), "rb")
		self.testLoss = pickle.load(f)
		f = open(os.path.join(self.dataDir, "testAcc.pkl"), "rb")
		self.testAcc = pickle.load(f)

	def draw(self):
		plt.figure(figsize=(8,4.5))
		ax1 = plt.subplot()
		plt.title("training and testing")
		ax1.set_xlabel("epochs")
		x = range(len(self.trainLoss))
		ax1.set_ylabel("loss")
		ax2 = ax1.twinx()
		ax2.set_ylabel("accuracy")

		kwargs={
			"marker":None,
			"lw":2,
			"alpha":0.9
		}
		l1, = ax1.plot(x, self.trainLoss, color="tab:blue", label="train loss", **kwargs)
		l2, = ax2.plot(x, self.trainAcc, color="tab:orange", label="train accuracy", **kwargs)
		l3, = ax1.plot(x, self.testLoss, color="tab:green", label="test loss", **kwargs)
		l4, = ax2.plot(x, self.testAcc, color="tab:red", label="test accuracy", **kwargs)

		plt.legend(handles=[l1, l2, l3, l4], loc="center right")
		# sv = plt.gcf()
		# sv.savefig(os.path.join(self.imgDir,"trainAndTest.eps"),format="eps",dpi=300)
		plt.show()

	def drawLoss(self):
		plt.figure()
		x = range(len(self.trainLoss))
		plt.title("Loss of training and testing")
		plt.scatter(x, self.trainLoss, label="train loss", alpha=0.5)
		plt.scatter(x, self.testLoss, label="test loss", alpha=0.5)
		plt.legend()
		plt.show()

	def drawAcc(self):
		plt.figure()
		x = range(len(self.trainAcc))
		plt.title("Loss of training and testing")
		plt.scatter(x, self.trainAcc, label="train loss", alpha=0.5)
		plt.scatter(x, self.testAcc, label="test loss", alpha=0.5)
		plt.legend()
		plt.show()


class ProbMap:
	def __init__(self, numClasses, path, groundTruth, index, height, width, trainIndex):
		self.map = np.zeros((1, numClasses))
		self.groundTruth = groundTruth
		self.index = index
		self.saveDir = path
		self.height = height
		self.width = width
		self.groundTruth = np.argmax(self.groundTruth, axis=1)
		self.trainIndex = trainIndex

	def addData(self, data):
		self.map = np.concatenate((self.map, data), axis=0)

	def finish(self):
		self.map = np.delete(self.map, (0), axis=0)

	def save(self):
		f = open(os.path.join(self.saveDir, "probmap.pkl"), "wb")
		pickle.dump(self.map, file=f)
		print("probmap saved!")

	def restore(self):
		f = open(os.path.join(self.saveDir, "probmap.pkl"), "rb")
		self.map = pickle.load(f)

	def drawGt(self):
		groundTruth = np.zeros(shape=(self.height, self.width))
		with tqdm(total=np.shape(self.groundTruth)[0], desc="processing gt") as pbar:
			for i in range(np.shape(self.groundTruth)[0]):
				index = self.index[i]
				h = index // self.height
				w = index % self.height
				groundTruth[h][w] += (self.groundTruth[i] + 1)
				pbar.update()

		plt.imshow(groundTruth,cmap="Set1_r")
		plt.colorbar()
		plt.show()

	def drawPredictedMap(self):
		pred = np.argmax(self.map, axis=1)
		probMap = np.zeros(shape=(self.height, self.width))
		with tqdm(total=np.shape(self.groundTruth)[0], desc="processing gt") as pbar:
			for i in range(np.shape(self.groundTruth)[0]):
				index = self.index[i]
				h = index // self.height
				w = index % self.height
				probMap[h][w] += (pred[i] + 1)
				pbar.update()

		plt.imshow(probMap)
		plt.colorbar()
		plt.show()

	def drawProbMap3D(self):
		pred = np.argmax(self.map, axis=1)
		probMap = np.zeros(shape=(self.height, self.width))
		with tqdm(total=np.shape(self.groundTruth)[0], desc="processing gt") as pbar:
			for i in range(np.shape(self.groundTruth)[0]):
				index = self.index[i]
				h = index // self.height
				w = index % self.height
				probMap[h][w] += (pred[i] + 1)
				pbar.update()
		x=np.arange(0,self.height,1)
		y=np.arange(0,self.width,1)
		x,y=np.meshgrid(x,y)
		fig=plt.figure()
		ax=Axes3D(fig)
		ax.plot_surface(x,y,probMap,rstride=1,cstride=1)
		# plt.imshow(probMap)
		# plt.colorbar()
		plt.show()

	def drawProbMap(self):
		probMap = np.zeros(shape=[self.height, self.width])
		for i in range(np.shape(self.groundTruth)[0]):
			index = self.index[i]
			h = index // self.height
			w = index % self.height
			probMap[h][w] += self.map[i][self.groundTruth[i]]

		plt.imshow(probMap)
		plt.colorbar()
		plt.show()

	def drawTrainMap(self):
		groundTruth = np.zeros(shape=(self.height, self.width))
		with tqdm(total=np.shape(self.groundTruth)[0], desc="processing gt") as pbar:
			for i in range(np.shape(self.groundTruth)[0]):
				index = self.index[i]
				h = index // self.height
				w = index % self.height
				groundTruth[h][w] += (self.groundTruth[i] + 1)
				pbar.update()

		trainMap = np.zeros(shape=[self.height, self.width])
		# print(np.shape(self.groundTruth)[0])
		for i in range(np.shape(self.trainIndex)[0]):
			index = self.trainIndex[i]
			h = index // self.height
			w = index % self.height
			trainMap[h][w] = 1

		for i in range(self.height):
			for j in range(self.width):
				trainMap[i][j] *= groundTruth[i][j]

		plt.imshow(trainMap)
		plt.colorbar()
		plt.show()

	def drawTestMap(self):
		groundTruth = np.zeros(shape=(self.height, self.width))
		with tqdm(total=np.shape(self.groundTruth)[0], desc="processing gt") as pbar:
			for i in range(np.shape(self.groundTruth)[0]):
				index = self.index[i]
				h = index // self.height
				w = index % self.height
				groundTruth[h][w] += (self.groundTruth[i] + 1)
				pbar.update()

		testMap = np.ones(shape=[self.height, self.width])
		# print(np.shape(self.groundTruth)[0])
		for i in range(np.shape(self.trainIndex)[0]):
			index = self.trainIndex[i]
			h = index // self.height
			w = index % self.height
			testMap[h][w] = 0

		for i in range(self.height):
			for j in range(self.width):
				testMap[i][j] *= groundTruth[i][j]

		plt.imshow(testMap)
		plt.colorbar()
		plt.show()


if __name__ == '__main__':
	trainProcess = TrainProcess(os.path.join(".", "save", "default"))
	trainProcess.restore()
	trainProcess.draw()
	exit(0)

	pathName = []
	pathName.append("./data/Indian_pines.mat")
	pathName.append("./data/Indian_pines_gt.mat")
	matName = []
	matName.append("indian_pines")
	matName.append("indian_pines_gt")
	print("using indian pines**************************")
	dataloader = DataLoader(pathName, matName, 9, 0.05, 1)
	allLabeledPatch, allLabeledSpectrum, allLabeledLabel, allLabeledIndex = dataloader.loadAllLabeledData()
	probMap = ProbMap(dataloader.numClasses, os.path.join(".", "save", "default", "data"),
					  allLabeledLabel, allLabeledIndex, dataloader.height, dataloader.width, dataloader.trainIndex)
	probMap.restore()
	# print(dataloader.numClasses)
	# probMap.drawGt()
	# probMap.drawPredictedMap()
	probMap.drawProbMap()
	# probMap.drawGt()
	# probMap.drawTestMap()
	# probMap.drawTrainMap()
