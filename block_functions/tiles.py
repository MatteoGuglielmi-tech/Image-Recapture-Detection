import numpy as np

def tiling(img,W):
	img = np.array(img)
	tiles = []
	for r in range(0,img.shape[0] - W, W):
		for c in range(0,img.shape[1] - W, W):
			window = img[r:r+W,c:c+W]
			window = np.array(window)
			tile = (window, r, c)
			tiles.append(tile)
	return tiles

def tilingAblock(img, r, c, W):
	window = img[r:r+W,c:c+W]
	window = np.array(window)
	tile = (window, r, c)
	return tile
