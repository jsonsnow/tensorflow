# _*_ coding: UTF-8 _*_

import numpy as np
import tensorflow as tf

# download mnist header wriete lib(55000 * 28 * 28)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('min_data',one_hot=True)

input_x = tf.placeholder(tf.float32, [None, 28 * 28])/255.
ouput_y = tf.placeholder(tf.int32, [None, 10]) # 
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

# collection 3000 datas from test data

test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]

conv1 = tf.layers.conv2d(
	inputs=input_x_images,
	filters=32,
	kernel_size=[5,5],
	strides=1,
	padding='same',
	activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(
	inputs=conv1,
	pool_size = [2,2],
	strides=2)

conv2 = tf.layers.conv2d(
	inputs=pool1,
	filters=64,
	kernel_size=[5,5],
	strides=1,
	padding='same',
	activation=tf.nn.relu) # shape [14,14, 64]

pool2 = tf.layers.max_pooling2d(
	inputs=conv2,
	pool_size=[2,2],
	strides = 2) #shape [7,7 64]

# pingt ang 
flat = tf.reshape(pool2, [-1,7 * 7 * 64]) # shape [7 * 7 * 64]

#1014 ge all join layer
dense = tf.layers.dense(inputs=flat, units=1024,
	activation=tf.nn.relu)

# dropout
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

# 10 ge shen jin yuan de  quan lian jie ceng

logits = tf.layers.dense(inputs=dropout, units=10) # shape [1, 1, 10]

#calu error (Cross entropy Softmax)
loss = tf.losses.softmax_cross_entropy(
	onehot_labels=ouput_y, 
	logits=logits)

# Adam 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# return(accuracy, update_op), create two var
accuracy = tf.metrics.accuracy(
	labels=tf.argmax(ouput_y, axis=1),
	predictions=tf.argmax(logits, axis=1))[1]

sess = tf.Session()

init = tf.group(tf.global_variables_initializer(), 
	tf.local_variables_initializer())
sess.run(init)

for i in range(20000):
	batch = mnist.train.next_batch(50)
	train_loss , train_op_ = sess.run([loss, train_op],
		{input_x:batch[0],ouput_y:batch[1]})
	if i % 100 == 0:
		test_accuracy = sess.run(accuracy,
			{input_x:test_x, ouput_y:test_y})
		print("Step=%d, Train loss=%.4f. [Test accuracy=%.2f]"
			% (i, train_loss, test_accuracy))

test_output = sess.run(logits, 
	{input_x: test_x[:20]})
inferenced_y = np.argmax(test_output, 1)
print(inferenced_y, 'Inferenced numbers') #