import numpy as np
import pandas as pd
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
		for i in range(self.populationSize):
			chromosomeS = random.sample(range(self.N), self.N)  # 附加码
			chromosomeX = random_integers(0, 1, self.N)  # 变量码
			chromosome = (chromosomeS, chromosomeX)
			population.append(chromosome)
		self.population = population
		return population

	def fitness(self):
		fitnessValue = list()
		bestFitnessValue = 0
		bestFitnessValueIndex = -1
		for i in range(self.populationSize):
			funcValue, wSum, decodeValue = self.decode(self.population[i])
			fitnessValue.append(funcValue)
			if bestFitnessValue < funcValue:
				bestFitnessValue = funcValue
				bestFitnessValueIndex = i
		self.bestFitnessValue.append(bestFitnessValue)
		self.bestIndividual.append(self.population[bestFitnessValueIndex])
		return fitnessValue

	def decode(self, individual):
		chromosomeS, chromosomeX = individual
		decodeValue = list()
		wSum = 0
		vSum = 0
		for i in range(self.N):
			if chromosomeX[i] == 0:
				decodeValue.append(0)
			else:
				if wSum + self.W[chromosomeS[i]] <= self.Capacity:
					wSum += self.W[chromosomeS[i]]
					vSum += self.V[chromosomeS[i]]
					decodeValue.append(1)
				else:
					decodeValue.append(0)
		return vSum, wSum, decodeValue

	def selection(self, fitness):
		"""
		轮盘赌选择
		:param fitness:
		:return:
		"""
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
				crossoverPos = random.sample(range(0, self.N - 1), 2)
				(crossoverPosStart, crossoverPosEnd) = (min(crossoverPos), max(crossoverPos))
				child1, child2 = self.PMX((self.population[count*2], self.population[count*2+1]), crossoverPosStart, crossoverPosEnd)
				newPopulation.append(child1)
				newPopulation.append(child2)
			else:
				newPopulation.append(self.population[2 * count])
				newPopulation.append(self.population[2 * count + 1])
			count += 1
		self.population = newPopulation
		return newPopulation

	def PMX(self, individual, crossoverPosStart, crossoverPosEnd):
		"""
		PMX算法 部分匹配交叉
		:param individual:
		:param crossoverPosStart:
		:param crossoverPosEnd:
		:return: child1, child2
		"""
		chromosome1, chromosome2 = copy(individual)
		chromosome1S, chromosome1X = chromosome1
		chromosome2S, chromosome2X = chromosome2
		chromosome1SPart = chromosome1S[crossoverPosStart:crossoverPosEnd+1].tolist()
		chromosome2SPart = chromosome2S[crossoverPosStart:crossoverPosEnd+1].tolist()
		# 交叉的片段直接交换
		chromosome1S[crossoverPosStart:crossoverPosEnd+1] = chromosome2SPart
		chromosome2S[crossoverPosStart:crossoverPosEnd+1] = chromosome1SPart
		# 调整顺序，解决冲突
		for i in range(0, self.N):
			if i in range(crossoverPosStart, crossoverPosEnd+1):
				continue
			while chromosome1S[i] in chromosome2SPart:
				chromosome1S[i] = chromosome1SPart[chromosome2SPart.index(chromosome1S[i])]
			while chromosome2S[i] in chromosome1SPart:
				chromosome2S[i] = chromosome2SPart[chromosome1SPart.index(chromosome2S[i])]
		chromosome1X = [individual[0][1][j] for i in range(self.N) for j in range(self.N) if chromosome1S[i] == individual[0][0][j]]
		chromosome2X = [individual[1][1][j] for i in range(self.N) for j in range(self.N) if chromosome2S[i] == individual[1][0][j]]
		return (chromosome1S, chromosome1X), (chromosome2S, chromosome2X)

	def mutation(self):
		"""
		采用逆位遗传算子
		:return:
		"""
		mp = []
		for i in range(self.populationSize):
			mp.append(random.random())
		for i in range(self.populationSize):
			if mp[i] < self.mp:
				mutationPos = random.sample(range(1, self.N), 2)  # 第一个位置倒序切片会出错，所以第一个位置不切片了
				mutationPosStart = min(mutationPos)
				mutationPosEnd = max(mutationPos)
				self.population[i][0][mutationPosStart:mutationPosEnd] = self.population[i][0][mutationPosEnd-1:mutationPosStart-1:-1]
		return self.population

	def generation(self):
		pop = self.populationInit()
		# print("初代：", pop)
		for i in range(self.maxGeneration):
			fit = self.fitness()
			sel = self.selection(fit)
			# print("sel：", sel)
			cro = self.crossover()
			# print("cro：", cro)
			mut = self.mutation()
			vSum = self.decode(self.population[0])[0]
			if vSum < self.bestFitnessValue[i]:
				self.population[0] = self.bestIndividual[i]
			# print("mut：", mut)
			# print("第%d代：" % (i+1), self.population)

	def plot(self, xspan, bestFitnessValue):
		plt.figure(1)
		ax = plt.subplot(111)
		plt.title(u'适应度/迭代次数')
		ax.plot(range(xspan), bestFitnessValue, color='red', linewidth=1)
		plt.show()

#############################
# ###二重结构编码解决0-1背包## #
#############################


if __name__ == '__main__':
	print("目前最优值：%d\n----------------------" % pd.read_csv("./data/best10.tsv", header=None)[0])
	data = pd.read_csv("./data/data10.tsv", sep=' ', header=None)
	data = data.values
	capacity, n = data[0, :]
	w, v = data[1:, 0], data[1:, 1]
	knapsack = Knapsack(capacity, n, (w, v), populationSize=150, maxGeneration=400, cp=0.5, mp=0.01)
	start = time.clock()
	knapsack.generation()
	end = time.clock()
	bestValue = max(knapsack.bestFitnessValue)
	bestValueIndex = knapsack.bestFitnessValue.index(bestValue)
	bestIndividual = knapsack.bestIndividual[bestValueIndex]
	print("求解最佳值：", bestValue)
	print("装载量：%d/%d" % (knapsack.decode(bestIndividual)[1], capacity))
	print("最佳值出现的世代数：", bestValueIndex)
	print("最佳个体：", bestIndividual)
	print("求解耗时：", end-start)
	knapsack.plot(knapsack.maxGeneration, knapsack.bestFitnessValue)
