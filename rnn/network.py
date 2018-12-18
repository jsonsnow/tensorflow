# -*- coding: UTF-8 -*-

import tensorflow as tf

class Model(object):

	#构造函数
	def __init__(self, input_obj, is_training, hidden_size, vocab_size, num_layers, 
		dropout=0.5, init_scale=0.05):
		self.is_training = is_training
		self.input_obj = input_obj
		self.batch_size = input_obj.batch_size
		self.num_steps = input_obj.num_steps
		self.hidden_size = hidden_size

		with tf.device('/cpu:0'):
			#创建 词向量 （Word Embedding）, Embedding 表示 Dense Vector（密集向量）
			#词向量本质上是一种单词聚类(Clustering)的方法
			embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
			# embeding_lookup 返回词向量
			inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

		# 如果是 训练时 并且dropout 小于 1 ，是输入经过一个dropout层
		# dropout 防止过拟合
		if is_training and dropout < 1:
			inputs = tf.nn.dropout(inputs, dropout)

		# 状态(state)的存取和提取
		# 第二维 是 2 是因为对每一个LSTM 单元有两个来自上一单元的输入:
		# 一个是 前一时刻 LSTM 的输出 h(t-1)
		# 一个是 前一时刻的单元状态 c(t-1)
		# 这个c 和 h 是用于构建之后的 tf.contrib.rnn.LSTMStateTuple
		self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])


		# 每一层的状态
		state_per_layer_list = tf.unstack(self.init_state, axis=0)

		#初始的状态（包含前一时刻LSTM的输出h(t-1) 和前一时刻的单元状态c(t-1),用于之后的dynamic_rnn）
		rnn_tuple_state = tuple(
			[tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) 
			for idx in range(num_layers)])

		# 创建一个LSTM层， 其中的神经元数目是hidden_size 个(默认650 个)
		cell = tf.contrib.rnn.LSTMCell(hidden_size)

		if is_training and dropout < 1:
			cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)

		if num_layers > 1:
			cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

		output , self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, init_state=rnn_tuple_state)

		output = tf.reshape(output, [-1, hidden_size])

		softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))

		softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))

		logists = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

		logists = tf.reshape(logists, [self.batch_size, self.num_steps, vocab_size])

		loss = tf.contrib.seq2seq.sequence_loss(logists,
			self.input_obj.targets, #期望输出，形状默认为【20， 35】
			tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
			average_across_timesteps=False,
			average_across_batch=True)

		#更新代价(cost)
		self.cost = tf.reduce_sum(loss)

		# Softmax 算出来的概率
		self.softmax_out = tf.nn.softmax(tf.reshape(logists, [-1, vocab_size]))

		# 取最大概率的那个值作为预测
		self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)

		#预测值和真实值(目标)对比
		correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))

		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		if not is_training:
			return 

		self.learn_rate = tf.Variable(0.0, trainable=False)

		tvar = tf.trainable_variables()

		# tf.clip_by_gloal_nrom（实现 Gradient clipping （梯度裁剪））是为了防止梯度爆炸
		# tf.gradients 计算 self.clost对 tvars的梯度(求导)， 返回一个梯度的列表
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvar), 5)

		optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)

		self.train_op = optimizer.apply_gradients(
			zip(grads, tvar),
			global_step=tf.train.get_or_create_global_step())

		self.new_lr = tf.placeholder(tf.float32, shape=[])
		self.lr_update = tf.assign(self.learn_rate, self.new_lr)

	# 更新 学习率
	def assign_lr(self, session, lr_value):
		session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


