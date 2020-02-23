import numpy as np

LENGTH = None


def convertToOneHot(vector, num_classes=None):
	assert isinstance(vector, np.ndarray)
	# print(len(vector))
	assert len(vector) > 0

	if num_classes is None:
		num_classes = np.max(vector) + 1
	else:
		assert num_classes > 0
		assert num_classes >= np.max(vector)

	result = np.zeros(shape=(len(vector), num_classes))
	result[np.arange(len(vector)), vector] = 1
	return result.astype(int)


def calOA(probMap, groundTruth):
	pred = np.argmax(probMap, axis=1)
	groundTruth = np.argmax(groundTruth, axis=1)
	totalCorrect = np.sum(np.equal(pred, groundTruth))
	total = np.shape(groundTruth)[0]
	print("correct: %d, all: %d" % (totalCorrect, total))
	return totalCorrect.astype(float) / total

def selectData(DATA=1):
	if DATA == 1:
		pathName = []
		pathName.append("./data/Indian_pines.mat")
		pathName.append("./data/Indian_pines_gt.mat")
		matName = []
		matName.append("indian_pines")
		matName.append("indian_pines_gt")
	elif DATA == 2:
		pathName = []
		pathName.append("./data/PaviaU.mat")
		pathName.append("./data/PaviaU_gt.mat")
		matName = []
		matName.append("paviaU")
		matName.append("paviaU_gt")
	elif DATA == 3:
		pathName = []
		pathName.append("./data/Pavia.mat")
		pathName.append("./data/Pavia_gt.mat")
		matName = []
		matName.append("pavia")
		matName.append("pavia_gt")
	else:
		pathName = []
		pathName.append("./data/Indian_pines.mat")
		pathName.append("./data/Indian_pines_gt.mat")
		matName = []
		matName.append("indian_pines")
		matName.append("indian_pines_gt")

	return pathName,matName

if __name__=="__main__":
	import argparse
	parser=argparse.ArgumentParser()
	parser.add_argument("--foo",action="store_true")
	arg=parser.parse_args()

	print(arg.foo)