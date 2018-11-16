import pandas as pd
from pylab import *
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class TSP:
	def __init__(self, cityData, populationSize=10, maxGeneration=5, cp=0.8, mp=0.15):
		(self.Node, self.X, self.Y) = cityData
		self.population = []
		self.bestIndividual = []
		self.bestFitnessValue = []
		self.populationSize = populationSize
		self.maxGeneration = maxGeneration
		self.cp = cp
		self.mp = mp
		self.chromosomeLength = len(self.Node)

	def populationInit(self):
		"""
		初始化种群
		:return: population
		"""
		population = []
		pickList = range(0, self.chromosomeLength)
		for i in range(self.populationSize):
			# 随机生成单条染色体
			chromosome = list()
			chromosome.extend(random.sample(pickList, len(pickList)))
			population.append(chromosome)
		self.population = copy.deepcopy(population)
		return population

	def fitness(self):
		"""
		计算个体适应度，相当于最小化问题，取 1/distance
		:return:
		"""
		fitnessValue = []
		for i in range(self.populationSize):
			fitnessValue.append(1/self.sumValue(self.population[i]))
		bestFitnessValue = max(fitnessValue)
		bestFitnessIndex = fitnessValue.index(bestFitnessValue)
		bestIndividual = self.population[bestFitnessIndex]
		self.bestFitnessValue.append(bestFitnessValue)
		self.bestIndividual.append(bestIndividual)
		return fitnessValue

	def selection(self, fitnessValue):
		"""
		采用轮盘赌方式挑选个体
		# :param population:
		:param fitnessValue:
		:return: newPopulation
		"""
		newPopulation = []  # 由选择而产生的种群
		fitnessValueSum = sum(fitnessValue)
		selectionProbability = (np.array(fitnessValue)/fitnessValueSum).tolist()  # 选择概率
		accumulationProbability = []  # 累积概率
		probabilityTemp = 0  # 构建轮盘（斐波契数列）
		for i in selectionProbability:
			probabilityTemp += i
			accumulationProbability.append(probabilityTemp)
		accumulationProbability.append(1)
		# 生成选择个体的概率
		selectP = []
		for i in range(self.populationSize):
			selectP.append(random.random())
		selectP.sort()
		# 挑选个体
		sPos = 0  # 指向select列表的指针
		aPos = 0  # 指向accumulationProbability累积概率列表的指针
		while sPos < self.populationSize:
			if selectP[sPos] < accumulationProbability[aPos]:  # 选中
				newPopulation.append(self.population[aPos])
				sPos += 1
			else:
				aPos += 1
		self.population = copy.deepcopy(newPopulation)
		return newPopulation

	def crossover(self):
		"""
		交叉 采用PMX（部分匹配交叉）
		:return:
		"""
		newPopulation = []
		cp = []
		count = 0
		for i in range(int(self.populationSize/2)):
			cp.append(random.random())
		for cpi in cp:
			if cpi < self.cp:
				crossoverPos = random.sample(range(0, self.chromosomeLength-1), 2)
				(crossoverPosStart, crossoverPosEnd) = (min(crossoverPos), max(crossoverPos))
				(child1, child2), (father1, father2) = self.OX((self.population[count*2], self.population[count*2+1]), crossoverPosStart, crossoverPosEnd)
				newPopulation.append(child1)
				newPopulation.append(child2)
				# print("father1:", father1)
				# print("father2:", father2)
				# print("child1:", child1)
				# print("child2:", child2)
				# print("crossoverPosStart:", crossoverPosStart)
				# print("crossoverPosEnd:", crossoverPosEnd)
				# print("----------------------------")
			else:
				newPopulation.append(self.population[2*count])
				newPopulation.append(self.population[2*count+1])
			count += 1
		self.population = copy.deepcopy(newPopulation)
		return newPopulation

	def PMX(self, individual, crossoverPosStart, crossoverPosEnd):
		"""
		PMX算法 部分匹配交叉
		:param individual:
		:param crossoverPosStart:
		:param crossoverPosEnd:
		:return: child1, child2
		"""
		father = copy.deepcopy(individual)
		individual1Part = individual[0][crossoverPosStart:crossoverPosEnd+1]
		individual2Part = individual[1][crossoverPosStart:crossoverPosEnd+1]
		# 交叉的片段直接交换
		individual[0][crossoverPosStart:crossoverPosEnd+1] = individual2Part
		individual[1][crossoverPosStart:crossoverPosEnd+1] = individual1Part
		# 调整顺序，解决冲突
		for i in range(0, self.chromosomeLength):
			if i in range(crossoverPosStart, crossoverPosEnd+1):
				continue
			while individual[0][i] in individual2Part:
				individual[0][i] = individual1Part[individual2Part.index(individual[0][i])]
			while individual[1][i] in individual1Part:
				individual[1][i] = individual2Part[individual1Part.index(individual[1][i])]
		return individual, father

	def OX(self, individual, crossoverPosStart, crossoverPosEnd):
		"""
		OX算法 顺序交叉算法
		:param individual:
		:param crossoverPosStart:
		:param crossoverPosEnd:
		:return: child1, child2
		"""
		father = copy.deepcopy(individual)
		individual1Part = individual[0][crossoverPosStart:crossoverPosEnd + 1]
		individual2Part = individual[1][crossoverPosStart:crossoverPosEnd + 1]
		p1, p2 = copy.deepcopy(individual)
		p1 = p1[crossoverPosEnd + 1:] + p1[0:crossoverPosEnd + 1]
		p2 = p2[crossoverPosEnd + 1:] + p2[0:crossoverPosEnd + 1]
		for i in range(len(individual1Part)):
			p1.remove(individual2Part[i])
			p2.remove(individual1Part[i])
		child1 = p2[(self.chromosomeLength-crossoverPosEnd):] + individual1Part + p2[0:(self.chromosomeLength-crossoverPosEnd)]
		child2 = p1[(self.chromosomeLength-crossoverPosEnd):] + individual2Part + p1[0:(self.chromosomeLength-crossoverPosEnd)]
		return (child1, child2), father

	def mutation(self):
		"""
		变异
		:return:
		"""
		newPopulation = []
		mp = []
		for i in range(self.populationSize):
			mp.append(random.random())
		for i in range(self.populationSize):
			if mp[i] < self.mp:
				mutationPos = random.sample(range(0, self.chromosomeLength), 2)
				self.population[i][mutationPos[0]], self.population[i][mutationPos[1]] = self.population[i][mutationPos[1]], self.population[i][mutationPos[0]]
				newPopulation.append(self.population[i])
			else:
				newPopulation.append(self.population[i])
		return newPopulation

	def run(self):
		population = self.populationInit()
		# print('初代:')
		# print(population)
		for i in range(self.maxGeneration):
			fitness = self.fitness()
			selectP = self.selection(fitness)
			crossP = self.crossover()
			mutationP = self.mutation()

			# fitnessValue = []
			# for j in range(self.populationSize):
			# 	fitnessValue.append(1/self.sumValue(self.population[j]))
			# badIndex = fitnessValue.index(min(fitnessValue))
			# self.population[badIndex] = self.bestIndividual[i]
			if self.sumValue(self.population[0]) > self.sumValue(self.bestIndividual[i]):
				self.population[0] = self.bestIndividual[i]

			# print("第%d代" % (i+1))
			# print(self.population)
		# print('selP:', selectP)
		# print('croP:', crossP)
		# print('mutP:', mutationP)
		return 0

	def plot(self, xSpan, X, Y, bestFitnessValue):
		plt.figure(1)
		ax = plt.subplot(211)
		plt.title(u'城市坐标')
		ax.scatter(X, Y, color='blue')
		ax = plt.subplot(212)
		plt.title(u'最佳适应度/迭代次数')
		ax.plot(range(xSpan), bestFitnessValue, color='blue', linewidth=1)
		plt.show()

	def sumValue(self, chromosome):
		"""
		计算一条路径的总花费
		:param chromosome:
		:return: cost
		"""
		cost = 0
		for i in range(self.chromosomeLength-1):
			pointA = (self.X[chromosome[i]], self.Y[chromosome[i]])
			pointB = (self.X[chromosome[i+1]], self.Y[chromosome[i+1]])
			cost += self.distance(pointA, pointB)
		pointA = (self.X[chromosome[self.chromosomeLength-1]], self.Y[chromosome[self.chromosomeLength-1]])
		pointB = (self.X[chromosome[0]], self.Y[chromosome[0]])
		cost += self.distance(pointA, pointB)
		return cost

	def distance(self, pointA, pointB):
		"""
		计算两点间距离
		:param pointA:
		:param pointB:
		:return:
		"""
		return math.sqrt(math.pow(pointA[0] - pointB[0], 2) + math.pow(pointA[1] - pointB[1], 2))


if __name__ == '__main__':
	data = pd.read_csv("cityData20.csv", header=None)
	(Node, X, Y) = (data[0], data[1], data[2])
	tsp = TSP(cityData=(Node, X, Y), populationSize=300, maxGeneration=300, cp=0.45, mp=0.02)

	bestRoad = [0, 2, 11, 1, 8, 16, 5, 19, 12, 4, 15, 17, 6, 18, 14, 9, 7, 3, 10, 13]
	print('最优路径：')
	for ni in bestRoad:
		print(Node[ni], end="")
	print("\n")
	print(tsp.sumValue(bestRoad))

	start = time.clock()
	tsp.run()
	end = time.clock()
	print('-----------------------------\n求解路径：')
	bestIndex = tsp.bestFitnessValue.index(max(tsp.bestFitnessValue))
	bestChromosome = tsp.bestIndividual[bestIndex]
	ind = bestChromosome.index(0)
	bestChromosome = bestChromosome[ind:] + bestChromosome[0:ind]
	for ni in bestChromosome:
		print(tsp.Node[ni], "->", end="")
	print(tsp.Node[bestChromosome[0]], "\n")
	print(tsp.sumValue(tsp.bestIndividual[bestIndex]))
	print('出现最佳值的代数：', bestIndex)
	print('耗时：', (end-start))
	tsp.plot(tsp.maxGeneration, X, Y, tsp.bestFitnessValue)