# _*_ Coding: UTF-8 _*_

'''
使用方法
'''
import os
import sys
import arg
import tensorflow as tf
import numpy as np

data_path = ''
parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default=data_path)

#如果是Python3版本
Py3 = sys.version_info[0] == 3

# 加载所有数据，读取所有单词，把其转成唯一对应的整数值

def load_data():
	#三个数据集的路径
	train_path = os.path.join(data_path,'ptb.train.txt')

	word_to_id = build_vocab(train_path)

def generate_bathes(raw_data, batch_size, num_steps):
	#将数据转为 Tensor类型
	raw_data = tf.convert_to_tensor()

class grass:
	@classmethod
	def bearItsSeed():
		return True
class wind:
	def ShakeItsLeaves():
		return True
class everything:
	@classmethod
	def Fine():
		print('就十分美好')
class we:
	def __init__(self,girl='teacher li',boy='teacher jiang'):
		self.girl = girl
		self.boy = boy	
	def standingUp(self):
		print('我们站着不说话')
	def silent(self):
		print('不说话')

		