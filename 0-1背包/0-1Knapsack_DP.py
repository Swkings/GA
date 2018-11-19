import numpy as np
import pandas as pd
import copy
from pylab import *
import random
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':
	print("目前最优值：%d\n----------------------" % pd.read_csv("./data/best10.tsv", header=None)[0])
	data = pd.read_csv("./data/data10.tsv", sep=' ', header=None)
	data = data.values
	capacity, n = data[0, :]
	m = capacity
	w, v = data[1:, 0], data[1:, 1]
	Matrix = np.zeros(((n+1), (capacity+1)))
	for i in range(1, n+1):
		for j in range(1, capacity+1):
			if j < w[i-1]:
				Matrix[i][j] = Matrix[i-1][j]
			else:
				Matrix[i][j] = max(Matrix[i-1][j], Matrix[i-1][j-w[i-1]]+v[i-1])

	print(Matrix[n][capacity])