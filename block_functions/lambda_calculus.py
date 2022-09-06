import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps



def interpolation(arr, new_length):
    old_indices = np.arange(0, len(arr))
    new_indices = np.linspace(0, len(arr)-1, new_length)
    spl = UnivariateSpline(old_indices, arr, k=3, s=0)
    new_array = spl(new_indices)
    return new_array

def interpolate_block(block, int_factor):
    block = np.array(block)
    block_int = np.zeros([block.shape[0]*int_factor, block.shape[1]]) #S.shape[0]*4 due too interpolation;
    for c in range(block.shape[1]):
        qi = block[:, c]
        interpolated_qi = interpolation(qi, block_int.shape[0])
        block_int[:, c] = interpolated_qi
    return block_int

def interpolate_blocks(list, int_factor):
	list_int = []
	for x in list: 
		x = interpolate_block(x, int_factor)
		list_int.append(x)
	return list_int

def spectral_energy(arr):
	return simps(pow(arr, 2))

def position_calculus_start(w, center):
	if (center - w) > 0:
		start = center - w
	else:
		start = 0
	return start

def position_calculus_end(w, center, size):
	if (center + w) < size:
		end = center + w
	else:
		end = size
	return end	

def find_lambda(arr):
	Eq = spectral_energy(arr)
	center = np.argmax(arr)
	w = 1
	devCount=0
	arrSize=len(arr)
	start = position_calculus_start(w, center)
	end = position_calculus_end(w, center, arrSize)
	data_extracted = arr[start : end]
	Ew = spectral_energy(data_extracted)
	while Ew < 0.90*Eq and devCount < arrSize-3:
		devCount=devCount+1
		start = position_calculus_start(w, center)
		end = position_calculus_end(w, center, arrSize)
		data_extracted = arr[start : end]
		Ew = spectral_energy(data_extracted)
		w += 1
	return w 

def find_lambda_average(block): #work with columns
	block = np.array(block)
	lambda_sum = 0
	for c in range(block.shape[1]):
		col = block[ : , c]
		lambda_sum = lambda_sum + find_lambda(col)
	lambda_avg = lambda_sum/(block.shape[1]) 
	return lambda_avg
