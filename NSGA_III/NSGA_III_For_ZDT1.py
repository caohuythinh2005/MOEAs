import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import random

def findHyperPlaneEquation(point1, point2):
    A = np.array([point1, point2])
    B = np.ones(2)
    X = np.linalg.solve(A, B)
    return X

def getIntersection(point1, point2):
    getPara = findHyperPlaneEquation(point1, point2)
    x1 = [1 / getPara[0], 0]
    x2 = [0, 1 / getPara[1]]
    return np.array([x1[0], x2[1]])

def dominate(p1, p2):
    if (p1[1][0] <= p2[1][0] and p1[1][1] <= p2[1][1]):
        if (p1[1][0] < p2[1][0] or p1[1][1] < p2[1][1]):
            return True
        
    return False    

def fixPopulation(pop1):
    popDic = {}
    for i in range(0, len(pop1)):
        popDic[str(pop1[i])] = 0

    index = []
    for i in range(0, len(pop1)):
        if popDic[str(pop1[i])] == 0:
            popDic[str(pop1[i])] = 1
        else:
            index.append(i)

    newPop = []
    for i in range(0, len(pop1)):
        if i not in index:
            newPop.append(pop1[i])

    return newPop


class ZDT1:
    def __init__(self, sizePop = 200, size = 100, k = 100
                 , n = 2, crossover_rate = 0.9, mutation_rate = 0.01, n_c = 2, p = 50, generation = 20):
        self.sizePop = sizePop
        self.size = size
        self.k = k
        self.n = n
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.p = p
        self.n_c = n_c
        self.generation = generation
        self.ideal = []
        self.lstVectorChose = []
        self.population = []
        self.ReferencePoint = []
        pass

    def GeneratePopulation(self):
        for i in range(0, self.sizePop):
            coor = []
            for j in range(0, self.n):
                coor.append(random.random())

            fit = self.Fitness(coor)
            self.population.append([coor, fit])

    def Fitness(self, res):
        newSum = 0
        for i in range(1, self.n):
            newSum += res[i]

        g = 1 + 9 * (newSum) / (self.n - 1)
        f1 = res[0]
        f2 = g * (1 - np.sqrt(res[0] / g))
        return [f1, f2]

    def Normalize(self, FitnessSet):
        pop = np.array(FitnessSet)
        ideal = np.min(pop, axis = 0)
        self.ideal = ideal
        pop = pop - ideal
        weightMatrix = np.array([[0, 1e6], [1e6, 0]])
        lstVectorChoosen1 = []
        for i in range(0, 2):
            newVec = pop * weightMatrix[i]
            lstVectorChoosen1.append(pop[np.argmin(np.max(newVec, axis = 1))])

        ReferencePoint = []
        lstVectorChoosen = np.array(lstVectorChoosen1)
        self.lstVectorChose = lstVectorChoosen
        if (lstVectorChoosen[0][0] * lstVectorChoosen[1][1] - lstVectorChoosen[0][1] * lstVectorChoosen[1][0]) != 0:
            vecIntersec = getIntersection(lstVectorChoosen[0], lstVectorChoosen[1])
            for i in range(0, self.p + 1):
                #ReferencePoint.append([[(i / self.p) * vecIntersec[0], ((self.p - i) / self.p) * vecIntersec[1]], 0])
                ReferencePoint.append([[(i / self.p) * vecIntersec[0], ((self.p - i) / self.p) * vecIntersec[1]], 0])
        return ReferencePoint

    def AssociateOperation(self, population, ReferencePoint):
        AssignSet = []
        for x in population:
            distanceSet = []
            fitness = x[1]
            for y in ReferencePoint:
                c = np.abs((x[1][0] - self.ideal[0]) * y[0][1] - (x[1][1] - self.ideal[1]) * y[0][0])
                d = np.sqrt(y[0][0] * y[0][0] + y[0][1] * y[0][1])
                distanceSet.append([y, c / d])
            index = 0
            ans = distanceSet[0]
            for i in range(1, len(distanceSet)):
                if (distanceSet[i][1] < distanceSet[index][1]):
                    index = i
                    ans = distanceSet[i]

            ReferencePoint[index][1] += 1
            AssignSet.append([x, ans[0]])

        return AssignSet
    
    def Niching(self, K, Fl, FitnessSet, ReferencePoint):
        Pop = Fl.copy()
        Ref = ReferencePoint.copy()
        Ouput = []
        k = 0
        AssignSet = self.AssociateOperation(Pop, Ref)
        while k < K:
            Ref.sort(key = lambda coor : coor[1])
            indexJmin = 0
            tmp = indexJmin
            lstJmin = []
            while (tmp <= len(Ref) - 1 and Ref[tmp][1] == Ref[indexJmin][1]):
                lstJmin.append([Ref[tmp], tmp])
                tmp += 1

            Jmin = random.choice(lstJmin)
            I_j = []
            for x in AssignSet:
                if (x[0] in Pop):
                    if x[1][0] == Jmin[0][0]:
                        I_j.append(x[0])
            if (len(I_j) > 0):
                s = random.choice(I_j)
                Ouput.append(s)
                Ref[Jmin[1]][1] += 1
                Pop.remove(s)
                k = k + 1
            else:
                Ref.remove(Jmin[0])

        return Ouput
    
    def Mutate(self, p):
        newChild = []
        if (random.random() < self.mutation_rate):
            c = []
            a = random.randint(0, len(p[0]))
            b = random.randint(0, len(p[0]))
            c += p[0][:a]
            c += p[0][a : b][: : -1]
            c += p[0][b:]
            fit = self.Fitness(c)
            newChild.append(c)
            newChild.append(fit)

        return newChild

    def MutatePopulation(self):
        newPop = self.population.copy()
        for popMutate in self.population:
            c = self.Mutate(popMutate)
            if (len(c) > 0):
                newPop.append(c)
            
        self.population = newPop.copy()

    def fixPopulation(self):
        popDic = {}
        for i in range(0, len(self.population)):
            popDic[str(self.population[i])] = 0

        index = []
        for i in range(0, len(self.population)):
            if popDic[str(self.population[i])] == 0:
                popDic[str(self.population[i])] = 1
            else:
                index.append(i)

        newPop = []
        for i in range(0, len(self.population)):
            if i not in index:
                newPop.append(self.population[i])

        self.population = newPop.copy()

    def CrossOver(self, p1, p2):
        newLstChild = []
        child1 = []
        child2 = []
        isCross = random.random()
        if (isCross <= self.crossover_rate):
            Beta = 0
            u = random.random()
            if (u <= 0.5):
                Beta = (2 * u) ** (1 / (self.n_c + 1))
            else:
                Beta = (1 / (2 * (1 - u))) ** (1 / (self.n_c + 1))
            pop1 = []
            pop2 = []
            for i in range(0, self.n):
                c1 = 0.5 * ((1 + Beta) * p1[0][i] + (1 - Beta) * p2[0][i])
                c2 = 0.5 * ((1 - Beta) * p1[0][i] + (1 + Beta) * p2[0][i])
                d1 = min(max(c1, 0), 1)
                d2 = min(max(c2, 0), 1)
                pop1.append(d1)
                pop2.append(d2)


            fit1 = self.Fitness(pop1)
            fit2 = self.Fitness(pop2)
            child1.append(pop1)
            child1.append(fit1)
            child2.append(pop2)
            child2.append(fit2)
            newLstChild.append(child1)
            newLstChild.append(child2)
        
        return newLstChild

    def BreedPopulation(self):
        child = self.population.copy()
        for i in range(0, self.k):
            a = random.choice(self.population)
            b = random.choice(self.population)
            c = self.CrossOver(a, b)
            for j in range(0, len(c)):
                if (len(c[j]) > 0):
                    child.append(c[j])
        self.population = child.copy()

    def FastNonDominatedSort(self):
        listRank = []
        listDominate = {}
        Rank0 = []
        for p in self.population:
            listDominate[str(p)] = []
        for p in self.population:
            if len(listDominate[str(p)]) == 0:
                S = []
                n1 = 0
                for q in self.population:
                    if (dominate(q, p)):
                        n1 += 1
                    elif (dominate(p, q)):
                        S.append(q)

                if (n1 == 0):
                    Rank0.append(p)

                listDominate[str(p)] = [S, n1]

        listRank.append(Rank0)
        print(" luc luong Pareto la : " + str(len(Rank0)))
        i = 0
        while (len(listRank[i]) > 0):       
            Q = []
            for p in listRank[i]:
                for q in listDominate[str(p)][0]:
                    listDominate[str(q)][1] -= 1
                    if (listDominate[str(q)][1] == 0):
                        Q.append(q)
            i += 1
            listRank.append(Q)
        listRank.pop()
        return listRank

    def SelectPopulation(self):
        F = self.FastNonDominatedSort()
        nextGene = []
        GeneForNormalization = []
        i = 0
        while (True):
            GeneForNormalization += F[i]
            i += 1
            if (len(GeneForNormalization) >= self.size):
                break

        if (len(GeneForNormalization) == self.size):
            self.population = GeneForNormalization
        else :
            for s in range(0, i - 1):
                nextGene += F[s]

            K = self.size - len(nextGene)
            FitnessSet = []
            for j in range(0, len(GeneForNormalization)):
                FitnessSet.append(GeneForNormalization[j][1])

            ReferencePoint = self.Normalize(FitnessSet)
            Ouput = self.Niching(K, F[i - 1], FitnessSet, ReferencePoint)
            for x in Ouput:
                nextGene.append(x)

            self.population = nextGene.copy()

    def Run(self):
        for i in range(self.generation):
            print("the he thu : " + str(i))
            self.fixPopulation()
            self.SelectPopulation()
            self.BreedPopulation()
            self.MutatePopulation()

newProblem = ZDT1(400, 200, 200, 2, 0.9, 0.01, 2, 50, 20)
newProblem.GeneratePopulation()
newProblem.Run()


getRank = newProblem.FastNonDominatedSort()
lastParetor = []
for i in range(0, len(getRank[0])):
    lastParetor.append(getRank[0][i][1])



xvals0, yvals0 = zip(*lastParetor)
plt.plot(xvals0, yvals0, 'o',  color = 'red')
plt.show()




# pop = []
# for i in range(0, 100):\
#     x1 = random.random()
#     y1 = random.random()
#     pop.append([x1, y1])

# ReferencePoint = Normalize(pop, 5)
# xvals1, yvals1 = zip(*pop)
# plt.plot(xvals1, yvals1, 'o',  color = 'blue')
# xvals0, yvals0 = zip(*ReferencePoint)
# plt.plot(xvals0, yvals0, 'o',  color = 'red')
# plt.show()


# X = (a, b, c) va sieu phang cua ta la (a, b, c, 1)




#pop = pop / vecIntersec


# InVector1 = np.array([vecIntersec[0], 0])
# InVector2 = np.array([0, vecIntersec[1]])


# zeros_axis = np.zeros(1000)
# axis1 = np.linspace(0, vecIntersec[0], 1000)
# axis2 = np.linspace(0, vecIntersec[1], 1000)
# ideal = np.zeros(2)
# plt.plot(axis1, zeros_axis, 'o', color = 'green')
# plt.plot(zeros_axis, axis2, 'o', color = 'green')
# plt.plot(pop[:, 0], pop[:, 1], 'o')
# plt.plot([ideal[0]], [ideal[1]], 'o', color = 'green')
# plt.plot([nadir[0]], [nadir[1]], 'o', color = 'yellow')
# plt.plot(InVector1, InVector2, color = 'red')
# plt.show()

#ideal