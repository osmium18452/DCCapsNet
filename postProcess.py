import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from dataloader import DataLoader
from tqdm import tqdm


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
		ax1 = plt.subplot()
		plt.title("training and testing")
		ax1.set_xlabel("epochs")
		x = range(len(self.trainLoss))
		ax1.set_ylabel("loss")
		ax2 = ax1.twinx()
		ax2.set_ylabel("accuracy")

		l1, = ax1.plot(x, self.trainLoss, color="blue", label="train loss", alpha=0.5)
		l2, = ax2.plot(x, self.trainAcc, color="orange", label="train accuracy", alpha=0.5)
		l3, = ax1.plot(x, self.testLoss, color="green", label="test loss", alpha=0.5)
		l4, = ax2.plot(x, self.testAcc, color="red", label="test accuracy", alpha=0.5)

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
	def __init__(self, numClasses, path, groundTruth, index, height, width):
		self.map = np.zeros((1, numClasses))
		self.groundTruth = groundTruth
		self.index = index
		self.saveDir = path
		self.height = height
		self.width = width
		self.groundTruth = np.argmax(self.groundTruth, axis=1)

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
		# print(self.groundTruth)
		with tqdm(total=np.shape(self.groundTruth)[0], desc="processing gt") as pbar:
			for i in range(np.shape(self.groundTruth)[0]):
				index = self.index[i]
				h = index // self.height
				w = index % self.height
				groundTruth[h][w] += (self.groundTruth[i] + 1)
				pbar.update()

		plt.imshow(groundTruth)
		plt.colorbar()
		plt.show()

	def drawPm(self):
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
	dataloader = DataLoader(pathName, matName, 9, 0.5, 1)
	allLabeledPatch, allLabeledSpectrum, allLabeledLabel, allLabeledIndex = dataloader.loadAllLabeledData()
	probMap = ProbMap(dataloader.numClasses, os.path.join(".", "save", "default", "data"),
					  allLabeledLabel, allLabeledIndex, dataloader.height, dataloader.width)
	probMap.restore()
	probMap.drawGt()
	probMap.drawPm()
