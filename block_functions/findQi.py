import cv2
import numpy as np
from block_functions import tiles as tl
from block_functions import detection as detect
from block_functions import line_spread_profile as lsp
from block_functions import lambda_calculus as lc
from block_functions import selection_blocks as sb

#constant declaration
W = 16
B = 0.6


def findQi(path):
	#step 1: acquiring image
	img = cv2.imread(path) 
	#step 2: convert image in grayscale
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#step 3: canny edge detection
	gray_img_edges = cv2.Canny(gray_img,100,200)
	#step 4: dividing edge image into blocks
	npGray_img_edges = np.array(gray_img_edges)
	tiles = tl.tiling(npGray_img_edges, W)
	#step 5: detection of edge blocks with a sufficiently ong edge
	tilesFilteredOrz = []
	tilesFilteredVert = []
	tilesFilteredOrz = detect.detection(tiles, B, W, tilesFilteredOrz, 0) #search orizontal edges
	tilesFilteredVert = detect.detection(tiles, B, W, tilesFilteredVert, 1) #search vertical edges
	#step 6: find out the corresponding block in the grayscale image
	grayscale_tilesOrz = [] 
	grayscale_tilesVert = []
	for x in tilesFilteredOrz:
		grayscale_tilesOrz.append(tl.tilingAblock(gray_img, x[1], x[2], W))	
	for x in tilesFilteredVert:
		grayscale_tilesVert.append(tl.tilingAblock(gray_img, x[1], x[2], W))	
	#step 7: rotating blocks on order to obtain only orizontal edges
	for x in tilesFilteredVert: #step 7
		bt = np.transpose(x[0])
		xt = (bt, x[1], x[2])
		tilesFilteredOrz.append(xt)
	for x in grayscale_tilesVert:
		bt = np.transpose(x[0])
		xt = (bt, x[1], x[2])
		grayscale_tilesOrz.append(xt)
	#step 8: calculate the variance of blocks 
	variance_of_tiles = []
	for x in grayscale_tilesOrz:
		variance_of_tiles.append(x[0].var())
	#step 9: calculate the line_spread functions for each column of each block
	line_spread_functions_of_tiles = [] 
	for x in grayscale_tilesOrz:
		line_spread_functions_of_tiles.append(lsp.line_spread_function_for_block(x[0]))
	#step 10: find out the average lambda for each block
	lambdas_of_tiles = [] 
	line_spread_functions_of_tiles = lc.interpolate_blocks(line_spread_functions_of_tiles, 4)
	for x in line_spread_functions_of_tiles: 
		lambdas_of_tiles.append(lc.find_lambda_average(x))
	#step 11: put togheter informations
	grayTales_with_lambda_and_var = list(zip(grayscale_tilesOrz, lambdas_of_tiles, variance_of_tiles, line_spread_functions_of_tiles))
	#step 12: select best blocks according lambda and variance
	selected_blocks = sb.best_blocks(grayTales_with_lambda_and_var)
	print("selected blocks len: ", len(selected_blocks))
	#step 13: calculate the line_spread_matrix Qi
	Qi = np.array([])
	if len(selected_blocks) > 0:
		Qi = np.array(sb.qiMatrixFormation(selected_blocks))
	return Qi
