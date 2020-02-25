import tensorflow as tf
import capslayer as cl
import numpy as np


def DCCapsNet(patch, spectrum, k, output):
	pt = tf.layers.conv2d(
		patch,
		filters=50,
		kernel_size=3,
		strides=1,
		padding="same",
		activation=tf.nn.relu
	)
	pt = tf.layers.max_pooling2d(pt, 2, strides=2, padding="same")
	pt = tf.layers.conv2d(
		pt,
		filters=300,
		kernel_size=3,
		padding="same",
		activation=tf.nn.relu
	)
	pt = tf.layers.max_pooling2d(pt, 2, strides=2, padding="same")

	pt=tf.nn.dropout(pt,k)

	pt, ptAct = cl.layers.primaryCaps(
		pt,
		filters=64,
		kernel_size=3,
		strides=1,
		out_caps_dims=[8, 1],
		method="logistic",
		name="pt"
	)

	ptNumInput = np.prod(cl.shape(pt)[1:4])
	pt = tf.reshape(pt, shape=[-1, ptNumInput, 8, 1])
	ptAct = tf.reshape(ptAct, shape=[-1, ptNumInput])

	sp = tf.layers.conv1d(spectrum, filters=30, kernel_size=5, strides=1, padding="valid", activation=tf.nn.relu)
	sp = tf.layers.max_pooling1d(sp, 3, strides=2, padding="valid")
	sp = tf.layers.conv1d(sp, filters=40, kernel_size=5, strides=1, padding="valid", activation=tf.nn.relu)
	sp = tf.layers.max_pooling1d(sp, 3, strides=2, padding="valid")
	sp=tf.nn.dropout(sp,k)

	# print(sp.shape,"******************************")
	sp = tf.reshape(sp, [-1, sp.shape[1], 1, sp.shape[2]])

	sp, spAct = cl.layers.primaryCaps(
		sp,
		filters=64,
		kernel_size=(3, 1),
		strides=1,
		out_caps_dims=[8, 1],
		method="logistic",
		name="sp"
	)

	spNumInput = np.prod(cl.shape(sp)[1:4])
	sp = tf.reshape(sp, shape=[-1, spNumInput, 8, 1])
	spAct = tf.reshape(spAct, shape=[-1, spNumInput])

	net=tf.concat([pt,sp],1)
	act=tf.concat([ptAct,spAct],1)

	net,act=cl.layers.dense(
		net,act,
		num_outputs=output,
		out_caps_dims=[16, 1],
		routing_method="DynamicRouting"
	)
	return act



def CapsNet(net, output):
	conv1 = tf.layers.conv2d(
		net,
		filters=100,
		kernel_size=3,
		strides=1,
		padding="same",
		activation=tf.nn.relu,
		name="convLayer"
	)
	conv1 = tf.layers.max_pooling2d(conv1, 2, strides=2, padding="same")

	conv2 = tf.layers.conv2d(
		conv1,
		filters=300,
		kernel_size=3,
		padding="same",
		activation=tf.nn.relu
	)
	conv2 = tf.layers.max_pooling2d(conv2, 2, strides=2, padding="same")

	convCaps, activation = cl.layers.primaryCaps(
		conv2,
		filters=64,
		kernel_size=3,
		strides=1,
		out_caps_dims=[8, 1],
		method="logistic"
	)

	n_input = np.prod(cl.shape(convCaps)[1:4])
	convCaps = tf.reshape(convCaps, shape=[-1, n_input, 8, 1])
	activation = tf.reshape(activation, shape=[-1, n_input])

	rt_poses, rt_probs = cl.layers.dense(
		convCaps,
		activation,
		num_outputs=output,
		out_caps_dims=[16, 1],
		routing_method="DynamicRouting"
	)
	# print(rt_probs.shape)
	return rt_probs

def DCCN2(patch, spectrum, k, output):
	pt = tf.layers.conv2d(
		patch,
		filters=50,
		kernel_size=3,
		strides=1,
		padding="same",
		activation=tf.nn.relu
	)
	pt = tf.layers.max_pooling2d(pt, 2, strides=2, padding="same")

	pt=tf.nn.dropout(pt,k)

	pt, ptAct = cl.layers.primaryCaps(
		pt,
		filters=64,
		kernel_size=3,
		strides=1,
		out_caps_dims=[8, 1],
		method="logistic",
		name="pt"
	)

	ptNumInput = np.prod(cl.shape(pt)[1:4])
	pt = tf.reshape(pt, shape=[-1, ptNumInput, 8, 1])
	ptAct = tf.reshape(ptAct, shape=[-1, ptNumInput])

	sp = tf.layers.conv1d(spectrum, filters=30, kernel_size=7, strides=1, padding="valid", activation=tf.nn.relu)
	sp = tf.layers.max_pooling1d(sp, 7, strides=2, padding="valid")
	sp=tf.nn.dropout(sp,k)

	# print(sp.shape,"******************************")
	sp = tf.reshape(sp, [-1, sp.shape[1], 1, sp.shape[2]])

	sp, spAct = cl.layers.primaryCaps(
		sp,
		filters=64,
		kernel_size=(3, 1),
		strides=1,
		out_caps_dims=[8, 1],
		method="logistic",
		name="sp"
	)

	spNumInput = np.prod(cl.shape(sp)[1:4])
	sp = tf.reshape(sp, shape=[-1, spNumInput, 8, 1])
	spAct = tf.reshape(spAct, shape=[-1, spNumInput])

	net=tf.concat([pt,sp],1)
	act=tf.concat([ptAct,spAct],1)

	net,act=cl.layers.dense(
		net,act,
		num_outputs=output,
		out_caps_dims=[16, 1],
		routing_method="DynamicRouting"
	)
	return act

def DCCN3(patch, spectrum, k, output):
	pt = tf.layers.conv2d(
		patch,
		filters=50,
		kernel_size=3,
		strides=1,
		padding="same",
		activation=tf.nn.relu
	)
	pt=tf.nn.dropout(pt,k)

	pt, ptAct = cl.layers.primaryCaps(
		pt,
		filters=64,
		kernel_size=3,
		strides=1,
		out_caps_dims=[8, 1],
		method="logistic",
		name="pt"
	)

	ptNumInput = np.prod(cl.shape(pt)[1:4])
	pt = tf.reshape(pt, shape=[-1, ptNumInput, 8, 1])
	ptAct = tf.reshape(ptAct, shape=[-1, ptNumInput])

	sp = tf.layers.conv1d(spectrum, filters=30, kernel_size=7, strides=1, padding="valid", activation=tf.nn.relu)
	sp=tf.nn.dropout(sp,k)

	sp = tf.reshape(sp, [-1, sp.shape[1], 1, sp.shape[2]])

	sp, spAct = cl.layers.primaryCaps(
		sp,
		filters=64,
		kernel_size=(3, 1),
		strides=1,
		out_caps_dims=[6, 1],
		method="logistic",
		name="sp"
	)

	spNumInput = np.prod(cl.shape(sp)[1:4])
	sp = tf.reshape(sp, shape=[-1, spNumInput, 6, 1])
	spAct = tf.reshape(spAct, shape=[-1, spNumInput])

	net=tf.concat([pt,sp],1)
	act=tf.concat([ptAct,spAct],1)

	net,act=cl.layers.dense(
		net,act,
		num_outputs=output,
		out_caps_dims=[8, 1],
		routing_method="DynamicRouting"
	)
	return act