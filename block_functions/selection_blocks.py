import numpy as np
import math

def sort_lambda(arr):
	return sorted(arr, key=lambda item: (item[1]))

def sort_variance(arr):
	return sorted(arr, key=lambda item: (item[2]), reverse=True)

def best_blocks(arr):
	arr_sorted_lambda = sort_lambda(arr)
	range_for_lambda = int(0.1*len(arr))
	best_according_lambdas = arr_sorted_lambda[ : range_for_lambda] 

	arr_sorted_variance = sort_variance(arr)
	range_for_variance = int(0.2*len(arr))
	best_according_variance = arr_sorted_variance[ : range_for_variance]

	best_according_variance = sort_lambda(best_according_variance)
	intersect = []
	for x in best_according_lambdas:
		blockA = x[0]
		for y in best_according_variance:
			blockB = y[0]
			if blockA[1] == blockB[1] and blockA[2] == blockB[2]:
				intersect.append(x)
				break
	return intersect

def qiMatrixFormation(selected_blocks):
	Qi = selected_blocks[0][3] #Qi appartiene a R(W*M)
	for x in selected_blocks[1 :]: #salto il primo perchè già in Qi
		qi_block = np.array(x[3])
		Qi = np.c_[Qi, qi_block]
	return Qi

def reduce_S(S, red_factor):
    S = np.array(S)
    S_reduced = np.zeros([S.shape[0], math.floor(S.shape[1]/red_factor)]) #S.shape[1]/4 because we want to conserve 1 column over 4
    count = 0
    for c in range(S_reduced.shape[1]):
        qi = S[:, count]
        S_reduced[:, c] = qi
        count = count + 4
    return S_reduced