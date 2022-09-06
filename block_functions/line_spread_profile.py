import numpy as np

#def derivate_old(arr):
#	#HighPassFilter = [-1, 1]
#	HighPassFilter = [-1, 0, 1]
#	arr = np.array(arr)
#	derivate = np.convolve(arr, HighPassFilter, 'same')
#	return derivate

def derivate(arr):
	arr = np.array(arr)
	derivate = np.gradient(arr, 1)
	return derivate

def normalize(arr):
	norm = np.linalg.norm(arr)
	if norm == 0: 
		return arr
	return arr / norm

def line_spread_function_for_block(block): #work with columns
	block = np.array(block)
	block_line_spread_funct = np.zeros([block.shape[0],block.shape[1]])
	for c in range(block.shape[1]):
		yi = block[:, c]
		yi1 = derivate(yi)
		qi = normalize(yi1)
		block_line_spread_funct[:, c] = qi
	return block_line_spread_funct