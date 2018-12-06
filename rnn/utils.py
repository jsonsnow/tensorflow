# _*_ conding: UTF-8 _*_

import os
import sys
import datetime
import argparse
import collections

import numpy as np
#import tensorflow as tf

data_path = "/home/snow/Desktop/tensorflow/tensorflow/simple-examples/data"

a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print([ [row[col] for row in a ] for col in range(len(row))])


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=data_path)
args = parser.parse_args()

#if py3
Py3 = sys.version_info[0] == 3

#将文件根据语句结束标识符<eos>分割

def read_words(filename):
	with tf.gfile.GFile(filename, "r") as f:
		if py3:
			return f.read().replace("\n","<eos>").split()
		else:
			return f.read().decode("utf-8").replace("\n","<eos>").split()

#构造从单词到唯一整数值的映射

def build_vocab():
	data = read_words(filename)

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x:(-x[1],x[0]))
	words, _ = list(zip(*count_pairs))

	#单词到整数的映射
	word_to_id = dict(zip(words, range(len(words))))

	return word_to_id

def file_to_word_ids(filename, word_to_id):
	data = read_words(filename)
	
#记载所有数据，读取所有单词，把其转成唯一对应的
def load_data():
	train_path = os.path.join(data_path, 'ptb.train.txt')
	valid_path = os.path.join(data_path, 'ptb.valid.txt')
	test_path = os.path.join(data_path, 'ptb.test.txt')

	#建立词汇表，将所有单词(word)转为唯一对应的整数
	word_to_id = build_vocab(train_path)

	#
	vocab_size = len(word_to_id)

	#反转一个词汇表：为了之后从整数 转为 单词
	id_to_word = dict(zip(word_to_id.values()),)

