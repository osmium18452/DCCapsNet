import numpy as np
import scipy.io
from tqdm import tqdm
from utils import convertToOneHot, LENGTH
import time
import random
import scipy.ndimage


class DataLoader:
	classPatches, classSpectrum, classIndex = [], [], []
	allPatch, allPatchLabel, allSpectrum = [], [], []
	trainPatch, trainLabel, trainSpectrum = [], [], []
	testPatch, testLabel, testSpectrum = [], [], []
	numEachClass = []
	trainNum = 0
	testNum = 0
	allLabeledNum = 0

	def __init__(self, pathName, matName, patchSize, portionOrNum, ratio):
		# load data
		self.data = scipy.io.loadmat(pathName[0])[matName[0]]
		self.label = scipy.io.loadmat(pathName[1])[matName[1]]

		# prepare some basic propertities
		self.patchSize = patchSize
		self.numClasses = len(np.unique(self.label)) - 1
		self.height = self.data.shape[0]
		self.width = self.data.shape[1]
		self.bands = self.data.shape[2]

		for i in range(self.numClasses):
			self.classPatches.append([])
			self.classSpectrum.append([])
			self.classIndex.append([])

		# normalize and pad
		self.data = self.data.astype(float)
		for band in range(self.bands):
			# print(np.min(self.data[:,:,band]))
			self.data[:, :, band] = (self.data[:, :, band] - np.min(self.data[:, :, band])) / \
									(np.max(self.data[:, :, band]) - np.min(self.data[:, :, band]))
		padSize = patchSize // 2
		# print(np.shape(self.data))
		self.data = np.pad(self.data, ((padSize, padSize), (padSize, padSize), (0, 0)), "symmetric")
		# print(np.shape(self.data))
		# print(self.height,self.width,self.bands)

		self.__slice()
		with open("seeData.txt","w+") as f:
			last=np.shape(self.allPatch[0])
			for a,i in enumerate(self.allPatch):
				if last!=np.shape(i) :
					print(a,"last: ",last,"i: ",np.shape(i))
				last=np.shape(i)
		if portionOrNum < 1:
			self.__prepareDataByPortion(portionOrNum)
		else:
			self.__prepareDataByNum(portionOrNum)
		if ratio != 0:
			self.dataAugment(ratio)

		self.trainLabel = np.array(self.trainLabel)
		self.trainPatch = np.array(self.trainPatch)
		self.trainSpectrum = np.array(self.trainSpectrum)
		self.trainSpectrum = np.reshape(self.trainSpectrum, [-1, self.bands, 1])
		self.testLabel = np.array(self.testLabel)
		self.testPatch = np.array(self.testPatch)
		self.testSpectrum = np.array(self.testSpectrum)
		self.testSpectrum = np.reshape(self.testSpectrum, [-1, self.bands, 1])

		print(np.shape(self.trainLabel))
		self.trainLabel = convertToOneHot(self.trainLabel, num_classes=self.numClasses)
		self.testLabel = convertToOneHot(self.testLabel, num_classes=self.numClasses)
		self.trainNum = self.trainLabel.shape[0]
		self.testNum = self.testLabel.shape[0]

		for i in range(self.numClasses):
			self.allLabeledNum += self.numEachClass[i]

	def __patch(self, i, j):
		heightSlice = slice(i, i + self.patchSize)
		widthSlice = slice(j, j + self.patchSize)
		return self.data[heightSlice, widthSlice, :]

	def __slice(self):
		unique=np.unique(self.label)
		lut=np.zeros(np.max(unique)+1,dtype=np.int)
		for iter,i in enumerate(unique):
			lut[i]=iter
		self.label=lut[self.label]
		with tqdm(total=self.height * self.width, desc="slicing ", ncols=LENGTH) as pbar:
			for i in range(self.height):
				for j in range(self.width):
					tmpLabel = self.label[i, j]
					tmpSpectrum = self.data[i, j, :]
					tmpPatch = self.__patch(i, j)
					self.allPatchLabel.append(tmpLabel)
					self.allPatch.append(tmpPatch)
					self.allSpectrum.append(tmpSpectrum)
					if tmpLabel != 0:
						self.classPatches[tmpLabel - 1].append(tmpPatch)
						self.classSpectrum[tmpLabel - 1].append(tmpSpectrum)
						self.classIndex[tmpLabel - 1].append(i * self.height + j)
					pbar.update(1)
		# self.numEachClass.append(0)
		for i in range(self.numClasses):
			self.numEachClass.append(len(self.classIndex[i]))

	def __prepareDataByPortion(self, portion):
		np.random.seed(0)
		with tqdm(total=self.numClasses, desc="dividing", ncols=LENGTH) as pbar:
			for i in range(self.numClasses):
				label = i
				index = np.random.choice(self.numEachClass[label], int((self.numEachClass[label]) * portion + 0.5),
										 replace=False)
				self.trainPatch.extend(self.classPatches[label][j] for j in index)
				self.trainSpectrum.extend(self.classSpectrum[label][j] for j in index)
				self.trainLabel.extend(label for j in range(len(index)))
				self.trainNum += len(index)

				index = np.setdiff1d(range(self.numEachClass[label]), index)
				self.testLabel.extend(label for j in range(len(index)))
				self.testPatch.extend(self.classPatches[label][j] for j in index)
				self.testSpectrum.extend(self.classSpectrum[label][j] for j in index)
				self.testNum += len(index)

				pbar.update(1)

	def __prepareDataByNum(self, num):
		np.random.seed(0)
		with tqdm(total=self.numClasses, desc="dividing patches", ncols=LENGTH) as pbar:
			for i in range(self.numClasses):
				label = i
				index = np.random.choice(self.numEachClass[label], int(num), replace=False)
				self.trainPatch.extend(self.classPatches[label][j] for j in index)
				self.trainSpectrum.extend(self.classSpectrum[label][j] for j in index)
				self.trainLabel.extend(label for j in range(len(index)))
				self.trainNum += len(index)

				index = np.setdiff1d(range(self.numEachClass[label]), index)
				self.testLabel.extend(label for j in range(len(index)))
				self.testPatch.extend(self.classPatches[label][j] for j in index)
				self.testSpectrum.extend(self.classSpectrum[label][j] for j in index)
				self.testNum += len(index)

				pbar.update(1)

	def dataAugment(self, times):
		trainNums = self.trainNum
		with tqdm(total=int((times-1)*trainNums), desc="augmenting", ncols=LENGTH) as pbar:
			for i in range(int(times - 1)):
				for k in range(trainNums):
					j=random.randint(0,trainNums-1)
					augPatch = self.trainPatch[j]
					augSpec = self.trainSpectrum[j]
					augLabel=self.trainLabel[j]
					chg = random.randint(0,3)
					if chg == 0:
						augPatch = np.flipud(augPatch)
					elif chg == 1:
						augPatch = np.fliplr(augPatch)
					elif chg == 2:
						augPatch = augPatch + np.random.normal(0, 0.01, size=np.shape(augPatch))
						augSpec = augSpec + np.random.normal(0,0.01,size=np.shape(augSpec))
					else:
						angel = random.randrange(-180, 180, 30)
						augPatch = scipy.ndimage.interpolation.rotate(augPatch, angel, axes=(1, 0),
																	  reshape=False, output=None, order=3,
																	  mode='constant', cval=0.0, prefilter=False)
					self.trainPatch.append(augPatch)
					self.trainSpectrum.append(augSpec)
					self.trainLabel.append(augLabel)
					pbar.set_postfix(trainsformNo=chg)
					pbar.update(1)

	def __dataAugment(self, times):
		index = np.random.choice(range(self.trainNum), int(self.trainNum * times), replace=False)
		udPatch, udLabel, udSpectrum = [], [], []
		lrPatch, lrLabel, lrSpectrum = [], [], []
		noisePatch, noiseLabel, noiseSpectrum = [], [], []
		# angelPatch, angelLabel, angelSpectrum = [], [], []
		with tqdm(total=len(index), desc="augmenting", ncols=LENGTH) as pbar:
			for i in index:
				udPatch.append(np.flipud(self.trainPatch[i]))
				udSpectrum.append(self.trainSpectrum[i])
				udLabel.append(self.trainLabel[i])

				lrPatch.append(np.fliplr(self.trainPatch[i]))
				lrSpectrum.append(self.trainSpectrum[i])
				lrLabel.append(self.trainLabel[i])

				noisePatch.append(self.trainPatch[i] + np.random.normal(0, 0.01, size=np.shape(self.trainPatch[0])))
				noiseSpectrum.append(self.trainSpectrum[i])
				noiseLabel.append(self.trainLabel[i])

				# angel = random.randrange(-180, 180, 30)
				# angelPatch.append(scipy.ndimage.interpolation.rotate(self.trainPatch[i], angel, axes=(1, 0),
				# 													 reshape=False, output=None, order=3,
				# 													 mode='constant', cval=0.0, prefilter=False))
				# angelSpectrum.append(self.trainSpectrum[i])
				# angelLabel.append(self.trainLabel[i])

				pbar.update(1)
		# print(np.shape(self.trainPatch),type(self.trainPatch))
		self.trainPatch.extend(udPatch[i] for i in range(len(index)))
		self.trainSpectrum.extend(udSpectrum[i] for i in range(len(index)))
		self.trainLabel.extend(udLabel[i] for i in range(len(index)))

		self.trainPatch.extend(lrPatch[i] for i in range(len(index)))
		self.trainSpectrum.extend(lrSpectrum[i] for i in range(len(index)))
		self.trainLabel.extend(lrLabel[i] for i in range(len(index)))

		# print(np.shape(noisePatch), type(noisePatch))
		self.tra466159inPatch.extend(noisePatch[i] for i in range(len(index)))
		self.trainSpectrum.extend(noiseSpectrum[i] for i in range(len(index)))
		self.trainLabel.extend(noiseLabel[i] for i in range(len(index)))

	# print(np.shape(angelPatch), type(angelPatch))
	# self.trainPatch.extend(angelPatch[i] for i in range(len(index)))
	# self.trainSpectrum.extend(angelSpectrum[i] for i in range(len(index)))
	# self.trainLabel.extend(angelLabel[i] for i in range(len(index)))

	def loadTrainData(self):
		return self.trainPatch, self.trainSpectrum, self.trainLabel

	def loadTestData(self):
		return self.testPatch, self.testSpectrum, self.testLabel

	def loadAllPatch(self):
		return self.allPatch, self.allSpectrum, self.allPatchLabel

	def loadTrainPatchOnly(self):
		return self.trainPatch, self.trainLabel

	def loadTestPatchOnly(self):
		return self.testPatch, self.testLabel

	def loadAllPatchOnly(self):
		return self.allPatch, self.allPatchLabel

	def loadAllLabeledData(self, patchOnly=False):
		patch = []
		spectrum = []
		label = []
		index = []
		for i in range(self.numClasses):
			patch.extend(self.classPatches[i][j] for j in range(self.numEachClass[i]))
			spectrum.extend(self.classSpectrum[i][j] for j in range(self.numEachClass[i]))
			index.extend(self.classIndex[i][j] for j in range(self.numEachClass[i]))
			label.extend(i for j in range(self.numEachClass[i]))
		patch = np.array(patch)
		spectrum = np.array(spectrum)
		label = convertToOneHot(np.array(label))
		index=np.array(index)
		spectrum = np.reshape(spectrum, [-1, self.bands, 1])
		if patchOnly:
			return patch, label, index
		else:
			return patch, spectrum, label, index


if __name__ == "__main__":
	pathName = []
	pathName.append("./data/PaviaU.mat")
	pathName.append("./data/PaviaU_gt.mat")
	matName = []
	matName.append("paviaU")
	matName.append("paviaU_gt")
	# print(np.random.permutation(10))
	# print([5 for i in range(10)])

	data = DataLoader(pathName, matName, 5, 0.2, 0)
	patch, spectrum, label = data.loadAllLabeledData()
	print(np.shape(patch), np.shape(spectrum), np.shape(label))
# print(np.shape(patch), np.shape(label))
