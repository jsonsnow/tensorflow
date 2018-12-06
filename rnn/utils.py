# _*_ conding: UTF-8 _*_

import os
import sys
import time

data_path = "/home/snow/Desktop/tensorflow/tensorflow/simple-examples/data"

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=data_path)
args = parser.parse_args()

#if py3
Py3 = sys.version_info[0] == 3

