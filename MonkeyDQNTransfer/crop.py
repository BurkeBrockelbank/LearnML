"""
This program crops out stuck portions of the data set.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/crop.py
"""

import os

for root, dirs, filenames in os.walk('./AIDATA'):
	print(root)