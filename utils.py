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
	elif DATA == 4:
		pathName = []
		pathName.append("./data/Salinas_corrected.mat")
		pathName.append("./data/Salinas_gt.mat")
		matName = []
		matName.append("salinas_corrected")
		matName.append("salinas_gt")
		print("using salinas**************************")
	elif DATA == 5:
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

	return pathName, matName


if __name__ == "__main__":
	with open("testscript/newModelTest.sh", "w+") as f:
		counter = 0
		for lr in (0.0001, 0.0005):
			for r in (0.05, 0.1):
				for p in (7, 9):
					for a in (1, 5):
						for epoch in (50, 100):
							for data in (1, 2, 3, 4, 5):
								print(
									"python ./train.py -g $1 -b 50 -l %.4f -r %.2f -p %d -e %d -a %d -m 4 -d %d --use_best_model "
									"-d ./save/feb25/dataImg/%d --model_path ./save/feb25/model/%d" % (
									lr, r, p, epoch, a, data, counter, counter), file=f)
								counter += 1
						# print(epoch,lr,r,a,p,data)

	exit(0)

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--foo", action="store_true")
	arg = parser.parse_args()

	print(arg.foo)
