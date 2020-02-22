import matplotlib.pyplot as plt
import numpy as np


class TrainProcess:
	def __init__(self):
		self.trainLoss = np.array([])
		self.trainAcc = np.array([])
		self.testLoss = np.array([])
		self.testAcc = np.array([])

	def addData(self,trainLoss,trainAcc,testLoss,testAcc):
		self.trainLoss=np.append(self.trainLoss,trainLoss)
		self.trainAcc=np.append(self.trainAcc,trainAcc)
		self.testLoss=np.append(self.testLoss,testLoss)
		self.testAcc=np.append(self.testAcc,testAcc)

