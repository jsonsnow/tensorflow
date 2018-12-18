# -*- coding: UTF-8 -*-

from utils import *
from network import *

def test(model_path, test_data, vocab_size, id_to_word):
	test_input = Input(batch_size=20, num_steps=35, data=test_data)

	m = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocab_size, num_layers=2)

	saver = tf.train.Saver()

	with tf.Session() as sess:
		coord = tf.train.Coordinator()

		threads = tf.train.start_queue_runners(coord=coord)

		current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))

		saver.restore(sess, model_path)

		num_acc_batches = 30

		acc_check_thresh = 5

		check_batch_idx = 25

		accuracy = 0

		for batch in range(num_acc_batches):
			if batch == check_batch_idx:

			else:
				acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
		if batch >= acc_check_thresh:
			accuracy += acc

		print('平均精度:{:.3f}'.format(accuracy / (num_acc_batches - acc_check_thresh)))

		coord.request_stop()
		coord.join(threads)