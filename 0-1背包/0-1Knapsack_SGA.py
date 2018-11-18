import numpy as np
import pandas as pd
import copy
from pylab import *
import random
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Knapsack:
	def __init__(self, Capacity, N,  WV, populationSize=50, maxGeneration=20, cp=0.8, mp=0.001):
		"""
		初始化参数
		:param Capacity: 背包容量
		:param N: 物品个数
		:param WV: W：物品重量  V：物品价值
		:param populationSize: 种群大小
		:param maxGeneration: 繁衍世代
		:param cp: 交叉率
		:param mp: 变异率
		"""
		self.Capacity = Capacity
		self.N = N
		self.W, self.V = WV
		self.price = self.V/self.W  # 性价比
		self.populationSize = populationSize
		self.maxGeneration = maxGeneration
		self.cp = cp
		self.mp = mp
		self.population = list()
		self.bestFitnessValue = list()
		self.bestIndividual = list()

	def populationInit(self):
		population = list()
		isFull = False
		count = 0
		while not isFull:
			chromosome = random_integers(0, 1, self.N)
			# chromosome = self.greedyFix(chromosome)
			if self.check(chromosome):
				continue
			else:
				population.append(chromosome)
				count += 1
			if count >= self.populationSize:
				isFull = True
		self.population = population
		return population

	def check(self, individual):
		if sum(individual*self.W) > self.Capacity:
			# return True  # 只生成符合约束的个体, 但是生成初代种群太耗费时间了，可以直接用greedyFix修正个体
			return False  # 生成包括不符合约束的个体
		return False

	def fitness(self):
		fitnessValue = list()
		bestFitnessValue = 0
		bestFitnessValueIndex = -1
		for i in range(self.populationSize):
			funcValue = sum(self.greedyFix(self.population[i])*self.V)
			if funcValue > bestFitnessValue:
				bestFitnessValue = funcValue
				bestFitnessValueIndex = i
			fitnessValue.append(funcValue)
		self.bestFitnessValue.append(bestFitnessValue)
		self.bestIndividual.append(self.population[bestFitnessValueIndex])
		return fitnessValue

	def greedyFix(self, individual):
		"""
		在超出容量的背包中，逐渐剔除性价比最小的物品，直至达到约束要求
		:param individual:
		:return:
		"""
		while sum(individual * self.W) > self.Capacity:
			price = individual * self.price
			price = [item + Inf if item == 0 else item for item in price]
			individual[price.index(min(price))] = 0
		return individual

	def selection(self, fitness):
		"""
		轮盘赌选择 这次采用 np.random.choice()函数
		:param fitness:
		:return:
		"""
		# np.random.choice()函数只适用一维数据结构，所以采用选择下标的方式间接选择个体
		newPopulation = []
		for item in np.random.choice(a=range(self.populationSize), size=self.populationSize, replace=True, p=fitness/sum(fitness)):
			newPopulation.append(self.population[item])
		self.population = newPopulation
		return newPopulation

	def crossover(self):
		newPopulation = []
		cp = []
		count = 0
		for i in range(int(self.populationSize / 2)):
			cp.append(random.random())
		for cpi in cp:
			if cpi < self.cp:
				crossoverPos = random.randint(1, self.N - 2)  # 第一个位置和最后一个位置不交叉
				# 矩阵拼接用np.hstack()
				child1 = np.hstack(((self.population[2*count][0:crossoverPos]), (self.population[2*count+1][crossoverPos:self.N])))
				child2 = np.hstack(((self.population[2*count+1][0:crossoverPos]), (self.population[2*count][crossoverPos:self.N])))
				newPopulation.append(child1)
				newPopulation.append(child2)
			else:
				newPopulation.append(self.population[2 * count])
				newPopulation.append(self.population[2 * count + 1])
			count += 1
		self.population = newPopulation
		return newPopulation

	def mutation(self):
		mp = []
		for i in range(self.populationSize):
			mp.append(random.random())
		for i in range(self.populationSize):
			if mp[i] < self.mp:
				mutationPos = random.randint(0, self.N - 1)
				self.population[i][mutationPos] = self.population[i][mutationPos] ^ 1
		return self.population

	def generation(self):
		pop = self.populationInit()
		# print("初代：", pop)
		for i in range(self.maxGeneration):
			fit = self.fitness()
			self.selection(fit)
			self.crossover()
			self.mutation()
			# print("第%d代：" % (i+1), self.population)

	def plot(self, xspan, bestFitnessValue):
		plt.figure(1)
		ax = plt.subplot(111)
		plt.title(u'适应度/迭代次数')
		ax.plot(range(xspan), bestFitnessValue, color='red', linewidth=1)
		plt.show()


if __name__ == '__main__':
	print("目前最优值：%d\n----------------------" % pd.read_csv("./data/best10.tsv", header=None)[0])
	data = pd.read_csv("./data/data10.tsv", sep=' ', header=None)
	data = data.values
	capacity, n = data[0, :]
	w, v = data[1:, 0], data[1:, 1]
	knapsack = Knapsack(capacity, n, (w, v), populationSize=200, maxGeneration=150, cp=0.85, mp=0.05)
	start = time.clock()
	knapsack.generation()
	end = time.clock()
	bestValue = max(knapsack.bestFitnessValue)
	bestValueIndex = knapsack.bestFitnessValue.index(bestValue)
	bestIndividual = knapsack.bestIndividual[bestValueIndex]
	print("求解最佳值：", bestValue)
	print("装载量：%d/%d" % (sum(bestIndividual*w), capacity))
	print("最佳值出现的世代数：", bestValueIndex)
	print("最佳个体：", bestIndividual)
	print("求解耗时：", end-start)
	knapsack.plot(knapsack.maxGeneration, knapsack.bestFitnessValue)
