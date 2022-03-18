import os

def mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)