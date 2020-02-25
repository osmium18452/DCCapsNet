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
		print("using indian pines**************************")
	elif DATA == 2:
		pathName = []
		pathName.append("./data/PaviaU.mat")
		pathName.append("./data/PaviaU_gt.mat")
		matName = []
		matName.append("paviaU")
		matName.append("paviaU_gt")
		print("using pivia university**************************")
	elif DATA == 3:
		pathName = []
		pathName.append("./data/Pavia.mat")
		pathName.append("./data/Pavia_gt.mat")
		matName = []
		matName.append("pavia")
		matName.append("pavia_gt")
		print("using pavia city**************************")
	elif DATA==4:
		pathName = []
		pathName.append("./data/Salinas_corrected.mat")
		pathName.append("./data/Salinas_gt.mat")
		matName = []
		matName.append("salinas_corrected")
		matName.append("salinas_gt")
		print("using salinas**************************")
	elif DATA==5:
		pathName = []
		pathName.append("./data/SalinasA_corrected.mat")
		pathName.append("./data/SalinasA_gt.mat")
		matName = []
		matName.append("salinasA_corrected")
		matName.append("salinasA_gt")
		print("using salinasA**************************")
	else:
		pathName = []
		pathName.append("./data/Indian_pines.mat")
		pathName.append("./data/Indian_pines_gt.mat")
		matName = []
		matName.append("indian_pines")
		matName.append("indian_pines_gt")
		print("using indian pines**************************")

	return pathName,matName

if __name__=="__main__":
	import argparse
	parser=argparse.ArgumentParser()
	parser.add_argument("--foo",action="store_true")
	arg=parser.parse_args()

	print(arg.foo)