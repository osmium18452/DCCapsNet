import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


class TrainProcess:
	def __init__(self,dir):
		self.trainLoss = np.array([])
		self.trainAcc = np.array([])
		self.testLoss = np.array([])
		self.testAcc = np.array([])
		self.saveDirectory=dir

	def addData(self,trainLoss,trainAcc,testLoss,testAcc):
		self.trainLoss=np.append(self.trainLoss,trainLoss)
		self.trainAcc=np.append(self.trainAcc,trainAcc)
		self.testLoss=np.append(self.testLoss,testLoss)
		self.testAcc=np.append(self.testAcc,testAcc)

class ProbMap:
	def __init__(self,numClasses,path=None):
		if path==None:
			self.map=np.zeros((1, numClasses))
		else:
			f=open(path,"rb")
			self.map=pickle.load(f)

	def addData(self,data):
		self.map=np.concatenate((self.map,data),axis=0)

	def finish(self):
		self.map = np.delete(self.map, (0), axis=0)

	def save(self,path):
		f=open(os.path.join(path,"probmap.pkl"),"wb")
		pickle.dump(self.map,file=f)

	def restore(self,path):
		f=open(os.path.join(path,"probmap.pkl"),"rb")
		self.map=pickle.load(f)
