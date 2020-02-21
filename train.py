import tensorflow as tf
import numpy as np
from tqdm import tqdm
import capslayer as cl
import os
import argparse
from dataloader import DataLoader
from model import DCCapsNet, CapsNet
from utils import LENGTH

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", default=50, type=int)
parser.add_argument("-b", "--batch_size", default=100, type=int)
parser.add_argument("-l", "--lr", default=0.001, type=float)
parser.add_argument("-g", "--gpu", default="0")
parser.add_argument("-r", "--ratio", default=0.1, type=float)
parser.add_argument("-a", "--aug", default=1, type=float)
parser.add_argument("-p", "--patch_size", default=7, type=int)
parser.add_argument("-m", "--model", default=1, type=int)
parser.add_argument("-d", "--drop", default=1, type=float)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
EPOCHS = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
RATIO = args.ratio
AUGMENT_RATIO = args.aug
PATCH_SIZE = args.patch_size
DROP_OUT = args.drop

pathName = []
pathName.append("./data/Indian_pines.mat")
pathName.append("./data/Indian_pines_gt.mat")
matName = []
matName.append("indian_pines")
matName.append("indian_pines_gt")

dataloader = DataLoader(pathName, matName, PATCH_SIZE, RATIO, AUGMENT_RATIO)

# trainPatch, trainLabel = dataloader.loadTrainPatchOnly()
# testPatch, testLabel = dataloader.loadTestPatchOnly()

trainPatch, trainSpectrum, trainLabel = dataloader.loadTrainData()
testPatch, testSpectrum, testLabel = dataloader.loadTestData()
allLabeledPatch, allLabeledSpectrum, allLabeledLabel = dataloader.loadAllLabeledData()

# print(np.shape(trainPatch),np.shape(trainLabel))

w = tf.placeholder(shape=[None, dataloader.bands, 1], dtype=tf.float32)
x = tf.placeholder(shape=[None, dataloader.patchSize, dataloader.patchSize, dataloader.bands], dtype=tf.float32)
y = tf.placeholder(shape=[None, dataloader.numClasses], dtype=tf.float32)
k = tf.placeholder(dtype=tf.float32)

if args.model == 1:
	pred = DCCapsNet(x, w, k, dataloader.numClasses)
	print("USING DCCAPS***************************************")
else:
	pred = CapsNet(x, dataloader.numClasses)
	print("USING CAPS***************************************")
pred = tf.divide(pred, tf.reduce_sum(pred, 1, keep_dims=True))

loss = tf.reduce_mean(cl.losses.margin_loss(y, pred))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
correctPredictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPredictions, "float"))
init = tf.global_variables_initializer()
# print(np.shape(trainPatch), np.shape(trainSpectrum), np.shape(trainLabel))
# exit(0)

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(EPOCHS):
		if epoch % 5 == 0:
			permutation = np.random.permutation(trainPatch.shape[0])
			trainPatch = trainPatch[permutation, :, :, :]
			trainSpectrum = trainSpectrum[permutation, :]
			trainLabel = trainLabel[permutation, :]

		iter = dataloader.trainNum // BATCH_SIZE
		with tqdm(total=iter, desc="epoch %3d" % (epoch + 1), ncols=LENGTH) as pbar:
			for i in range(iter):
				batch_w = trainSpectrum[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :]
				batch_x = trainPatch[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :, :]
				batch_y = trainLabel[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :]
				_, batchLoss, trainAcc = sess.run([optimizer, loss, accuracy],
												  feed_dict={w: batch_w, x: batch_x, y: batch_y, k: DROP_OUT})
				pbar.set_postfix_str("loss: %.6f, accuracy:%.2f" % (batchLoss, trainAcc))
				pbar.update(1)

			if iter * BATCH_SIZE < dataloader.trainNum:
				batch_w = trainSpectrum[iter * BATCH_SIZE:, :, :]
				batch_x = trainPatch[iter * BATCH_SIZE:, :, :, :]
				batch_y = trainLabel[iter * BATCH_SIZE:, :]
				_, bl, ta = sess.run([optimizer, loss, accuracy],
									 feed_dict={w: batch_w, x: batch_x, y: batch_y, k: DROP_OUT})

			idx = np.random.choice(dataloader.testNum, size=BATCH_SIZE, replace=False)
			test_batch_w = testSpectrum[idx, :, :]
			test_batch_x = testPatch[idx, :, :, :]
			test_batch_y = testLabel[idx, :]
			ac, ls = sess.run([accuracy, loss], feed_dict={w: test_batch_w, x: test_batch_x, y: test_batch_y, k: 1})
			pbar.set_postfix_str(
				"loss: %.3f, accuracy:%.2f, testLoss:%.3f, testAcc:%.2f" % (batchLoss, trainAcc, ls, ac))

	iter = dataloader.allLabeledNum // BATCH_SIZE
	probMap=np.zeros((1,dataloader.numClasses))
	for i in range(iter):
		batch_w = trainSpectrum[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :]
		batch_x = trainPatch[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :, :]
		batch_y = trainLabel[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :]
		tmp=sess.run([pred],feed_dict={w: test_batch_w, x: test_batch_x, y: test_batch_y, k: 1})
		probMap=np.concatenate((probMap,tmp),axis=0)

	if iter*BATCH_SIZE < dataloader.allLabeledNum:
		batch_w = trainSpectrum[iter * BATCH_SIZE:, :, :]
		batch_x = trainPatch[iter * BATCH_SIZE:, :, :, :]
		batch_y = trainLabel[iter * BATCH_SIZE:, :]
		tmp = sess.run([pred], feed_dict={w: test_batch_w, x: test_batch_x, y: test_batch_y, k: 1})
		probMap=np.concatenate((probMap,tmp),axis=0)

	probMap = np.delete(probMap, (0), axis=0)
