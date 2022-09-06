import numpy as np

def detection(tiles, B, W, tilesFiltered, ax):
	for x in tiles:
		EdgeForCol = np.count_nonzero(x[0],axis=ax)
		oneEdgeCount = 0
		for y in range(W):
			if EdgeForCol[y] == 1:
				oneEdgeCount = oneEdgeCount + 1
		if oneEdgeCount >= B*W:
			tilesFiltered.append(x)
	return tilesFiltered

