import pandas as pd
from pylab import *
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class JSP:
	def __init__(self, WP, DP, PT, FN, ON, PP, populationSize=10, maxGeneration=5, cp=0.98, mp=0.01):
		self.WP, self.DP, self.PT, self.FN, self.ON, self.PP = WP, DP, PT, FN, ON, PP
		self.populationSize = populationSize
		self.maxGeneration = maxGeneration
		self.cp = cp
		self.mp = mp
		self.M = len(self.WP)
		self.N = max(self.DP)
		self.blankMatrix = np.zeros((self.M, self.N))  # 创建空白矩阵，方便后面使用
		self.population = list()
		self.bestFitness = list()
		self.bestIndividual = list()

	def populationInit(self):
		"""
		初始化种群
		:return: population
		"""
		population = []
		for i in range(self.populationSize):
			chromosome = self.createIndividual()
			population.append(chromosome)
			self.population = copy.deepcopy(population)
		return population

	def createIndividual(self):
		"""
		生成一个个体
		:return:
		"""
		# 生成单条染色体
		chromosome = copy.deepcopy(self.blankMatrix)
		# 对空白染色体进行处理，使之符合约束条件
		indexValues = self.DP.index.values
		for WPi in indexValues:
			series = self.randSeries(self.ON[WPi], self.FN[WPi], self.DP[WPi])
			for seriesPop in series:
				self.columnFix(chromosome, WPi, seriesPop)
		return chromosome

	def columnFix(self, chromosome, WPi, seriesPop):
		"""
		判断并修正列，使之符合约束
		:param chromosome:
		:param WPi:
		:param seriesPop:
		:return:
		"""
		temp = 0
		while True:
			WPij = random.randint(0, self.DP[WPi]-1)  # 矩阵从0开始
			bi = chromosome[:, WPij]
			# hi = self.PT.as_matrix()
			hi = self.PT.values
			if sum(bi * hi) + seriesPop*self.PT[WPi] <= 11 and chromosome[WPi][WPij] + seriesPop <= self.ON[WPi]:
				chromosome[WPi][WPij] += seriesPop
				return True
			else:
				temp += 1
				if temp > 100:
					chromosome[WPi][WPij] += seriesPop
					return False

	def randSeries(self, span, rSum, mLength):
		"""
		随机生成一个和为rSum,长度不超过mLength的一个序列
		:param span: 每个数的范围
		:param rSum: 序列和
		:param mLength: 序列最大长度
		:return:
		"""
		series = list()
		seriesSum = sum(series)
		seriesLength = len(series)
		while seriesSum != rSum:
			rn = random.randint(1, span)
			if seriesSum + rn > rSum:  # 保证序列和为rSum
				continue
			else:
				if seriesLength < mLength:   # 保证序列长度不超过mLength
					series.append(rn)
					seriesLength += 1
					seriesSum = sum(series)
				else:
					while True:   # 达到序列长度后，将生成的数加在序列的某个位置，保证不超过范围
						inx = random.randint(0, seriesLength-1)
						rn = random.randint(1, span-1)
						if series[inx] + rn <= span:
							series[inx] = series[inx] + rn
							seriesSum = sum(series)
							break
						else:
							pass
		return series

	def fitness(self):
		"""
		目标函数： sum(sum((Si-X[i])*(Si-X[i])))
		暂未考虑偏好参数Pi
		:return: fitnessValue
		"""
		fitnessValue = list()
		for i in range(self.populationSize):
			funcValue = 0
			chromosome = self.population[i]
			for wpi in range(self.M):
				isDP = True  # 找交工期
				Si = 0
				for dpj in reversed(range(self.DP[wpi])):
					if chromosome[wpi][dpj] != 0:
						if isDP:
							Si = dpj
							isDP = False  # 找到交工期
						else:
							funcValue += (Si - dpj)*(Si - dpj)
			fitnessValue.append(funcValue)
		bestFitness = min(fitnessValue)
		bestFitnessIndex = fitnessValue.index(bestFitness)
		bestIndividual = self.population[bestFitnessIndex]
		self.bestFitness.append(bestFitness)
		self.bestIndividual.append(bestIndividual)
		return fitnessValue

	def selection(self, fitnessValue):
		"""
		排序选择，排除20%劣质个体
		剩余80%全选，再将排在前20%选中
		:param fitnessValue:
		:return:
		"""
		newPopulation = list()
		# 将适应度序列化，待排序后可以根据其index找到对应个体
		fitnessValue = pd.Series(fitnessValue)
		fitnessValue = fitnessValue.sort_values(ascending=True)
		fitnessValueIndex = fitnessValue.index.values
		for i in range(round(len(fitnessValueIndex)*0.8)):
			newPopulation.append(self.population[fitnessValueIndex[i]])
		for i in range(round(len(fitnessValueIndex) * 0.2)):
			newPopulation.append(self.population[fitnessValueIndex[i]])
		self.population = copy.deepcopy(newPopulation)
		return newPopulation

	def crossover(self):
		"""
		单点交叉，由于经过选择操作后种群已经是有序的了，所以需要先乱序
		:return:
		"""
		newPopulation = []
		cp = []
		count = 0
		randIndex = random.sample(range(self.populationSize), self.populationSize)
		for i in range(int(self.populationSize/2)):
			cp.append(random.random())
		for cpi in cp:
			if cpi < self.cp:
				crossoverPos = random.randint(0, self.N)
				child1, child2 = self.SPX(self.population[randIndex[2 * count]], self.population[randIndex[2 * count + 1]], crossoverPos)
				newPopulation.append(child1)
				newPopulation.append(child2)
			else:
				newPopulation.append(self.population[randIndex[2 * count]])
				newPopulation.append(self.population[randIndex[2 * count + 1]])
			count += 1
		self.population = copy.deepcopy(newPopulation)
		return newPopulation

	def SPX(self, individual1, individual2, crossoverPos):
		"""
		单点交叉
		:param individual1:
		:param individual2:
		:param crossoverPos:
		:return: child1, child2
		"""
		# 注意，这里一定要深拷贝，否则交换的位置会互相引用，会导致一个个体不交换
		individual1Part = copy.deepcopy(individual1[:, crossoverPos:])
		individual2Part = copy.deepcopy(individual2[:, crossoverPos:])
		individual1[:, crossoverPos:] = copy.deepcopy(individual2Part)
		individual2[:, crossoverPos:] = copy.deepcopy(individual1Part)
		child1 = self.check(individual1)
		child2 = self.check(individual2)
		return child1, child2

	def check(self, individual):
		"""
		修正不合约束的个体，无法修正则被淘汰，新生个体取代
		:param individual:
		:return: newIndividual
		"""
		indexValues = self.DP.index.values
		for ind in indexValues:
			fn = sum(individual[ind, :])
			while fn != self.FN[ind]:
				if fn > self.FN[ind]:  # 如果夹具总数大于所需数，则随机挑选位置逐渐减少
					j = random.randint(0, self.DP[ind]-1)
					if individual[ind][j] > 0:
						individual[ind][j] -= 1
				else:
					randIndex = random.sample(range(self.DP[ind]), self.DP[ind])
					isModify = False  # 标记是否修正成功
					for j in randIndex:
						bi = individual[:, j]
						hi = self.PT.values
						if sum(bi * hi) + self.PT[ind] <= 11 and individual[ind][j] + 1 <= self.ON[ind]:
							individual[ind][j] += 1
							isModify = True
							break
					if not isModify:  # 如果修正不了，返回一个新个体
						return self.createIndividual()
				fn = sum(individual[ind, :])
		newIndividual = copy.deepcopy(individual)
		return newIndividual

	def mutation(self):
		"""
		变异
		:return: newPopulation
		"""
		newPopulation = list()
		mp = list()
		for i in range(self.populationSize):
			mp.append(random.random())
		for i in range(self.populationSize):
			if mp[i] < self.mp:
				mutAble = True  # 标记变异是否符合约束，一旦不符合，重新选择位置
				while mutAble:
					mutationPos = random.sample(range(0, self.N), 2)
					mutation1Part = copy.deepcopy(self.population[i][:, mutationPos[0]])
					mutation2Part = copy.deepcopy(self.population[i][:, mutationPos[1]])
					for j in range(self.M):
						if mutation1Part[j] > 0 and mutationPos[1] >= self.DP[j]:
							mutAble = False
							break
						if mutation2Part[j] > 0 and mutationPos[0] >= self.DP[j]:
							mutAble = False
							break
					if mutAble:
						self.population[i][:, mutationPos[0]] = copy.deepcopy(mutation1Part)
						self.population[i][:, mutationPos[1]] = copy.deepcopy(mutation2Part)
						newPopulation.append(self.population[i])
						break
					else:   # 不符合要求重新寻找
						mutAble = True
			else:
				newPopulation.append(self.population[i])
		self.population = copy.deepcopy(newPopulation)
		return newPopulation

	def plot(self, xSpan, bestFitness):
		plt.figure(1)
		ax = plt.subplot(211)
		plt.title(u'最佳适应度/迭代次数')
		ax.plot(range(xSpan+1), bestFitness, color='blue', linewidth=1)
		plt.show()

	def run(self):
		pop = self.populationInit()
		print('初代:')
		# print(pop)
		fit = self.fitness()  # 后面保留精英个体，替换劣质个体会重新计算下一代，放在循环外面节省时间
		for i in range(self.maxGeneration):
			sel = self.selection(fit)
			cro = self.crossover()
			mut = self.mutation()
			fit = self.fitness()
			badIndex = fit.index(max(fit))
			self.population[badIndex] = self.bestIndividual[i]
			fit[badIndex] = self.bestFitness[i]
			print("第%d代" % (i+1))
			# print(self.population)

#####################################
# Workpiece(WP):工件
# Delivery period(DP):交工期
# processing time(PT):加工时间（1个夹具）
# number of fixtures(FN):总夹具数
# ON : 一天可能使用夹具数
# Processing priority(PP):加工优先级
# X[i]:工件i加工日集合
# 目标函数： sum(sum((Si-X[i])*(Si-X[i])))
# 约束：
#     横向：sum（B[i][:]） = FN[i]
#     纵向：sum（B[:][j] * PT） <= 11
# 书上最优值：1320（种群：120， 世代：400， 交叉率：0.98， 变异率：0.01）， 但是书上给出的最优值个体中第11天列项不符合约束条件
# 但实际种群在200代左右就开始收敛了， 值在1400左右
####################################


if __name__ == "__main__":
	data = pd.read_csv("jspData10x23.csv")
	# data = data.sort_index(by=['DP'], ascending=False)
	# print(data)
	WP, DP, PT, FN, ON, PP = data['WP'], data['DP'], data['PT'], data['FN'], data['ON'], data['PP']
	DP = DP.sort_values(ascending=True)
	# inx = DP.index.values
	# # print(FN[inx[0]])
	jsp = JSP(WP, DP, PT, FN, ON, PP, populationSize=200, maxGeneration=200, cp=0.98, mp=0.01)
	start = time.clock()
	jsp.run()
	end = time.clock()
	bestFitnessValue = min(jsp.bestFitness)
	bestFitnessIndex = jsp.bestFitness.index(bestFitnessValue)
	bestIndividual = jsp.bestIndividual[bestFitnessIndex]
	bestIndividual = pd.DataFrame(bestIndividual, index=WP, columns=range(1, max(DP)+1))

	print(bestFitnessValue)
	print(bestIndividual)
	print('出现最佳值的代数：', bestFitnessIndex)
	print('耗时：', (end - start))
	jsp.plot(jsp.maxGeneration, jsp.bestFitness)