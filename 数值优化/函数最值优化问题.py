import random
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GA:
	def __init__(self, objFunc, genoNum, valueSpan, accuracy=6, populationSize=10, maxGeneration=5, cp=0.8, mp=0.2, isMin=True):
		"""
		初始化各参数
		:param objFunc: 目标函数
		:param genoNum: 每条染色体基因个数
		:param valueSpan: 取值范围
		:param accuracy: 要求的精度（结果保留的小数位数）
		:param populationSize: 种群大小
		:param cp: 交叉概率
		:param mp: 变异概率
		:param isMin: True最小化问题， False最大化问题
		"""
		self.population = []
		self.bestFitness = []
		self.bestFitnessValue = []
		self.bestFitnessParam = []
		self.objFunc = objFunc
		self.genoNum = genoNum
		self.minValue = valueSpan[0]
		self.maxValue = valueSpan[1]
		self.genotypeLength = math.ceil(math.log2((self.maxValue-self.minValue) * math.pow(10, accuracy)))  # 基因型长度
		self.chromosomeLength = self.genotypeLength * self.genoNum  # 染色体长度
		self.populationSize = populationSize
		self.maxGeneration = maxGeneration
		self.isMin = isMin
		self.cp = cp
		self.mp = mp

	def populationInit(self):
		"""
		初始化种群
		:return: population
		"""
		population = []
		for i in range(self.populationSize):
			# 随机生成单条染色体
			chromosome = []
			for j in range(self.chromosomeLength):
				chromosome.append(random.randint(0, 1))
			population.append(chromosome)
		self.population = copy.deepcopy(population)
		return population

	def fitness(self):
		"""
		计算种群中个体适应度
		# :param population: 种群
		:return: fitnessValue
		"""
		fitnessValueTemp, fitnessParamTemp = self.objFuncValue(self.population)
		fitnessValueTemp = np.array(fitnessValueTemp)
		if self.isMin:
			fitnessValueTemp = -fitnessValueTemp
		avgV = fitnessValueTemp.mean()
		minV = fitnessValueTemp.min()
		fitnessValue = (fitnessValueTemp + math.fabs(avgV) + math.fabs(minV)).tolist()  # 将所有的适应度转换成正值
		bestPos = fitnessValue.index(max(fitnessValue))
		self.bestFitness.append(self.population[bestPos])
		self.bestFitnessValue.append(max(fitnessValue))
		self.bestFitnessParam.append(fitnessParamTemp[bestPos])
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
		# 生成能存活个体的概率
		select = []
		for i in range(self.populationSize):
			select.append(random.random())
		select.sort()
		# 挑选个体
		sPos = 0  # 指向select列表的指针
		aPos = 0  # 指向accumulationProbability累积概率列表的指针
		while sPos < self.populationSize:
			if select[sPos] < accumulationProbability[aPos]:  # 选中
				newPopulation.append(self.population[aPos])
				sPos += 1
			else:
				aPos += 1
		self.population = copy.deepcopy(newPopulation)
		return newPopulation

	def crossover(self):
		"""
		交叉
		:return:
		"""
		newPopulation = []
		cp = []
		count = 0
		for i in range(int(self.populationSize/2)):
			cp.append(random.random())
		for cpi in cp:
			if cpi < self.cp:
				crossoverPos = random.randint(1, self.chromosomeLength-2)  # 第一个位置和最后一个位置不交叉
				child1 = (self.population[2*count][0:crossoverPos])+(self.population[2*count+1][crossoverPos:self.chromosomeLength])
				child2 = (self.population[2*count+1][0:crossoverPos])+(self.population[2*count][crossoverPos:self.chromosomeLength])
				newPopulation.append(child1)
				newPopulation.append(child2)
			else:
				newPopulation.append(self.population[2*count])
				newPopulation.append(self.population[2*count+1])
			count += 1
		self.population = copy.deepcopy(newPopulation)
		return newPopulation

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
				mutationPos = random.randint(0, self.chromosomeLength-1)
				self.population[i][mutationPos] = self.population[i][mutationPos] ^ 1
				newPopulation.append(self.population[i])
			else:
				newPopulation.append(self.population[i])
		return newPopulation

	def b2d(self, binary):
		"""
		二进制转成十进制
		:param binary: 需要转换的二进制串
		:return : decimal 十进制数
		"""
		decimal = 0
		for i in range(len(binary)):
			decimal += binary[i]*math.pow(2, i)
		return decimal

	def decode(self, binary):
		"""
		将二进制转换成实际对应的值
		:param binary: 二进制串
		:return: value
		"""
		decimal = self.b2d(binary)
		value = self.minValue + decimal * (self.maxValue-self.minValue)/(math.pow(2, self.genotypeLength)-1)
		return value

	def objFuncValue(self, genotypeList):
		"""
		计算函数值
		:param genotypeList: 参数列表
		:return: functionValue
		"""
		functionValue = []
		paramList = []
		for i in range(len(genotypeList)):
			param = []
			for j in range(self.genoNum):
				genotype = genotypeList[i][j * self.genotypeLength:(j + 1) * self.genotypeLength]
				param.append(self.decode(genotype))
			functionValue.append(self.objFunc(param))
			paramList.append(param)
		return functionValue, paramList

	def plot(self, xspan, funcValue, bestFitnessValue):
		plt.figure(1)
		ax = plt.subplot(211)
		plt.title(u'函数值/迭代次数')
		ax.plot(range(xspan), funcValue, color='red', linewidth=1)
		ax = plt.subplot(212)
		plt.title(u'最佳适应度/迭代次数')
		ax.plot(range(xspan), bestFitnessValue, color='blue', linewidth=1)
		plt.show()

	def run(self):
		population = self.populationInit()
		print('初代:')
		print(population)
		for i in range(self.maxGeneration):
			fitness = self.fitness()
			newP = self.selection(fitness)
			crossP = self.crossover()
			mutationP = self.mutation()
			# print("第%d代" % i)
			# print(self.population)
			# print('newP:', newP)
			# print('croP:', crossP)
			# print('mutP:', mutationP)
		return 0


if __name__ == '__main__':
	def objF1(param):
		return param[0]**2 + param[1]**2 + param[2]**2
	span1 = (-5.12, 5.12)
	paraNum1 = 3
	isMin1 = True

	def objF2(param):
		return 100 * math.pow(math.pow(param[0], 2)-param[1], 2) + math.pow(1 - param[0], 2)
	span2 = (-2.048, 2.048)
	paraNum2 = 2
	isMin2 = True

	def objF3(param):
		return param[0]*math.sin(10*math.pi*param[0]) + 2.0
	span3 = (-1, 2)
	paraNum3 = 1
	isMin3 = False

	def objF4(param):
		return (4-2.1*math.pow(param[0], 2)+(math.pow(param[0], 4)/3))*math.pow(param[0], 2) + param[0]*param[1] + (-4+4*math.pow(param[1], 2))*math.pow(param[1], 2)
	span4 = (-5, 5)
	paraNum4 = 2
	isMin4 = True

	GA = GA(objFunc=objF3, genoNum=paraNum3, valueSpan=span3, accuracy=6, cp=0.75, mp=0.2, populationSize=600, maxGeneration=200, isMin=isMin3)
	pop = GA.run()
	funcValue, param = GA.objFuncValue(GA.bestFitness)
	GA.plot(GA.maxGeneration, funcValue, GA.bestFitnessValue)
	print('每代中最佳适应值：', GA.bestFitnessValue)
	print('每代中最佳适应参数：', GA.bestFitnessParam)
	print('每代中最佳函数值：', funcValue)
	if isMin3:
		bestValue = min(funcValue)
	else:
		bestValue = max(funcValue)
	bestIndex = funcValue.index(bestValue)
	bestParam = param[bestIndex]
	print('最佳值：', bestValue)
	print('最佳参数：', bestParam)
	print('出现最佳值的代数：', bestIndex)