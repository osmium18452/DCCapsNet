import numpy as np

LENGTH=None
def convertToOneHot(vector, num_classes=None):

	assert isinstance(vector, np.ndarray)
	assert len(vector) > 0

	if num_classes is None:
		num_classes = np.max(vector) + 1
	else:
		assert num_classes > 0
		assert num_classes >= np.max(vector)

	result = np.zeros(shape=(len(vector), num_classes))
	result[np.arange(len(vector)), vector] = 1
	return result.astype(int)

def calOA(probMap,groundTruth):
	pred=np.argmax(probMap,axis=1)
	groundTruth=np.argmax(groundTruth,axis=1)
	totalCorrect=np.sum(np.equal(pred,groundTruth))
	total=np.shape(groundTruth)[0]
	print("true: %d, all: %d"%(totalCorrect,total))
	return totalCorrect.astype(float)/total


