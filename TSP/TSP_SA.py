#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Swking
@File  : TSP_SA.py
@Date  : 2018/11/29
@Desc  : 模拟退火
"""
import pandas as pd
from pylab import *
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TSP_SA:
	def __init__(self, cityData, sT=1000, eT=0.001, alpha=0.98, mapkob=10):
		(self.Node, self.X, self.Y) = cityData
		self.length = len(self.Node)
		self.alpha = alpha
		self.mapkob = mapkob
		self.sT = sT
		self.eT = eT
		self.bestFitnessValue = []
		self.bestIndividual = []

	def sumValue(self, individual):
		"""
		计算一条路径的总花费
		:param individual:
		:return: cost
		"""
		cost = 0
		for i in range(self.length-1):
			pointA = (self.X[individual[i]], self.Y[individual[i]])
			pointB = (self.X[individual[i+1]], self.Y[individual[i+1]])
			cost += self.distance(pointA, pointB)
		pointA = (self.X[individual[self.length-1]], self.Y[individual[self.length-1]])
		pointB = (self.X[individual[0]], self.Y[individual[0]])
		cost += self.distance(pointA, pointB)
		return cost

	def distance(self, pointA, pointB):
		return ceil(math.sqrt((math.pow(pointA[0]-pointB[0], 2) + math.pow(pointA[1]-pointB[1], 2))/10))

	def mutation(self, individual):
		mutationPos = random.sample(range(1, self.length), 2)
		posStart, posEnd = (min(mutationPos), max(mutationPos))
		if random.random() <= 0.5:
			individual[posStart], individual[posEnd] = individual[posEnd], individual[posStart]
		else:
			cutPart = copy.deepcopy(individual[posStart:posEnd+1])
			individual[posStart:posEnd+1] = cutPart[::-1]
		return individual

	def drawScatter(self):
		plt.scatter(self.X, self.Y)
		plt.show()

	def drawPath(self):
		cityX = []
		cityY = []
		bestFitness = min(self.bestFitnessValue)
		bestIndex = self.bestFitnessValue.index(bestFitness)
		bestIndividual = self.bestIndividual[bestIndex]
		bestIndividual.append(0)
		for i in bestIndividual:
			cityX.append(self.X[i])
			cityY.append(self.Y[i])
		plt.plot(cityX, cityY)
		plt.show()

	def drawCost(self):
		plt.plot(range(1, len(self.bestFitnessValue)+1), self.bestFitnessValue)
		plt.show()

	def run(self):
		# 初始化一条路径
		individual = list()
		individual.append(0)
		individual.extend(random.sample(range(1, self.length), self.length-1))
		fitness = self.sumValue(individual)
		self.bestIndividual.append(individual)
		self.bestFitnessValue.append(fitness)
		plt.ion()
		plt.figure(figsize=(5, 8))

		while self.eT < self.sT:
			for i in range(self.mapkob):
				newIndividual = copy.deepcopy(individual)
				newIndividual = self.mutation(newIndividual)
				newFitness = self.sumValue(newIndividual)

				DELTA = newFitness - fitness
				if DELTA < 0:
					individual = copy.deepcopy(newIndividual)
					fitness = newFitness
					# self.bestIndividual.append(individual)
					# self.bestFitnessValue.append(fitness)
				else:
					P = math.exp(-DELTA/self.sT)
					if random.random() < P:
						individual = copy.deepcopy(newIndividual)
						fitness = newFitness
						# self.bestIndividual.append(individual)
						# self.bestFitnessValue.append(fitness)
					else:
						pass
			self.sT *= self.alpha
			self.bestIndividual.append(individual)
			self.bestFitnessValue.append(fitness)
			plt.clf()
			cityX = copy.deepcopy([])
			cityY = copy.deepcopy([])
			for i in individual:
				cityX.append(self.X[i])
				cityY.append(self.Y[i])
			cityX.append(self.X[0])
			cityY.append(self.Y[0])
			plt.subplot(2, 1, 1)
			plt.plot(cityX, cityY, 'r-', label=fitness)
			plt.title('Path')
			plt.legend()
			# plt.draw()
			plt.subplot(2, 1, 2)
			plt.title('Cost')
			plt.plot(range(len(self.bestFitnessValue)), self.bestFitnessValue, label=self.sT)
			plt.legend()
			plt.xlim([0, 700])
			plt.ylim([10000, 50000])
			plt.pause(0.0000001)
		plt.ioff()
		# plt.show()
		return plt

if __name__ == '__main__':
	data = pd.read_csv("cityData48.csv", header=None)
	(Node, X, Y) = (data[0], data[1], data[2])

	# cityData20.csv
	# tsp = TSP_SA(cityData=(Node, X, Y), alpha=0.98, mapkob=10)
	# bestRoad = [0, 2, 11, 1, 8, 16, 5, 19, 12, 4, 15, 17, 6, 18, 14, 9, 7, 3, 10, 13]
	# print('最优路径：')
	# for ni in bestRoad:
	# 	print(Node[ni], end="->")
	# print(Node[0], end="")
	# print("\n", end="")
	# print(tsp.sumValue(bestRoad))

	# cityData20.csv
	tsp = TSP_SA(cityData=(Node, X, Y), sT=100, eT=0.1, alpha=0.99, mapkob=10)
	bestRoad = [1,8,38,31,44,18,7,28,6,37,19,27,17,43,30,36,46,33,20,47,21,32,39,48,5,42,24,10,45,35,4,26,2,29,34,41,16,22,3,23,14,25,13,11,12,15,40,9]
	bestRoad = np.array(bestRoad) - 1
	print('最优路径：')
	for ni in bestRoad:
		print(Node[ni], end="->")
	print(Node[0], end="")
	print("\n", end="")
	print(tsp.sumValue(bestRoad))


	start = time.clock()
	img = tsp.run()
	end = time.clock()


	bestFitness = min(tsp.bestFitnessValue)
	bestIndex = tsp.bestFitnessValue.index(bestFitness)
	bestIndividual = tsp.bestIndividual[bestIndex]
	print('\n求解路径：')
	for ni in bestIndividual:
		print(Node[ni], end="->")
	print(Node[0], end="")
	print("\n", end="")
	print(bestFitness)
	print('耗时：', (end - start))
	# tsp.drawCost()
	# tsp.drawScatter()
	# tsp.drawPath()
	img.show()