import numpy as np
import matplotlib.pyplot as plt
import random


INF = 9999999999


sizePop = 400
size = 100
k = 300
generation = 100
n = 2
crossover_rate = 0.9
mutation_rate = 0.01
n_c = 4

def Fitness(p):
    Msum = 0
    for i in range(1, n):
        Msum += p[i]

    g = 1 + 9 * (Msum) / (n - 1)
    f1 = p[0]
    f2 = g * (1 - np.sqrt(p[0] / g))
    return [f1, f2]

population = []

for i in range(0, sizePop):
    coor = []
    for j in range(0, n):
        coor.append(random.random())

    fit = Fitness(coor)
    population.append([coor, fit])


def dominate(p1, p2):
    if (p1[1][0] <= p2[1][0] and p1[1][1] <= p2[1][1]):
        if (p1[1][0] < p2[1][0] or p1[1][1] < p2[1][1]):
            return True
        
    return False    

def FastNonDominatedSort(pop):
    listRank = []
    listDominate = {}
    Rank0 = []
    for p in pop:
        listDominate[str(p)] = []
    for p in pop:
        if len(listDominate[str(p)]) == 0:
            S = []
            n1 = 0
            for q in pop:
                if (dominate(q, p)):
                    n1 += 1
                elif (dominate(p, q)):
                    S.append(q)

            if (n1 == 0):
                Rank0.append(p)

            listDominate[str(p)] = [S, n1]
        else :
            listDominate[str(p)][0] = 2 * listDominate[str(p)][0]
            listDominate[str(p)][1] = 2 * listDominate[str(p)][1]
    listRank.append(Rank0)
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
    k1 = 0
    return listRank



def CrowdingDistance(pop):
    l = len(pop)
    Ldistance = {}
    for i in range(0, l):
        Ldistance[str(pop[i])] = 0
    for i in range(0, 2):
        a = pop.copy()
        a.sort(key = lambda coor : coor[1][i])
        fmin = a[0][1][i]
        fmax = a[l - 1][1][i]
        Ldistance[str(a[0])] += INF
        Ldistance[str(a[l - 1])] += INF
        for j in range(1, l - 2):
            Ldistance[str(a[j])] += (a[j + 1][1][i] - a[j - 1][1][i])/(fmax - fmin)

    ans = []
    for j in range(0, l):
        newDis = []
        newDis.append(pop[j])
        newDis.append(Ldistance[str(pop[j])])
        ans.append(newDis)

    return ans

def SelectPopulation(P):
    F = FastNonDominatedSort(P)
    nextGene = []
    i = 0
    while (len(nextGene) + len(F[i])) <= size:
        nextGene += F[i]
        i += 1
    if i < len(F):
        lstCrow = CrowdingDistance(F[i])
        lstCrow.sort(key = lambda coor : coor[1], reverse=True)
        l = len(nextGene)
        for j in range(0, size - l):
            nextGene.append(lstCrow[j][0])
    return nextGene


def CrossOver(p1, p2):
    newLstChild = []
    child1 = []
    child2 = []
    isCross = random.random()
    if (isCross <= crossover_rate):
        Beta = 0
        u = random.random()
        if (u <= 0.5):
            Beta = (2 * u) ** (1 / (n_c + 1))
        else:
            Beta = (1 / (2 * (1 - u))) ** (1 / (n_c + 1))


        pop1 = []
        pop2 = []
        for i in range(0, n):
            c1 = 0.5 * ((1 + Beta) * p1[0][i] + (1 - Beta) * p2[0][i])
            c2 = 0.5 * ((1 - Beta) * p1[0][i] + (1 + Beta) * p2[0][i])
            d1 = min(max(c1, 0), 1)
            d2 = min(max(c2, 0), 1)
            pop1.append(d1)
            pop2.append(d2)


        fit1 = Fitness(pop1)
        fit2 = Fitness(pop2)
        child1.append(pop1)
        child1.append(fit1)
        child2.append(pop2)
        child2.append(fit2)
        newLstChild.append(child1)
        newLstChild.append(child2)
        
    return newLstChild
    

def BreedPopulation(pop):
    child = pop.copy()
    for i in range(0, k):
        a = random.choice(pop)
        b = random.choice(pop)
        c = CrossOver(a, b)
        for j in range(0, len(c)):
            if (len(c[j]) > 0):
                child.append(c[j])
    return child


def Mutate(p):
    newChild = []
    if (random.random() < mutation_rate):
        c = []
        a = random.randint(0, len(p[0]))
        b = random.randint(0, len(p[0]))
        c += p[0][:a]
        c += p[0][a : b][: : -1]
        c += p[0][b:]
        fit = Fitness(c)
        newChild.append(c)
        newChild.append(fit)

    return newChild


def MutatePopulation(pop):
    newPop = pop.copy()
    for popMutate in pop:
        c = Mutate(popMutate)
        if (len(c) > 0):
            newPop.append(c)
        
    return newPop


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
for i in range(generation):
    print("the he thu " + str(i))
    population = fixPopulation(population)
    matingPool = SelectPopulation(population)
    children = BreedPopulation(matingPool)
    population = MutatePopulation(children)


getRank = FastNonDominatedSort(population)


lastParetor = []
for i in range(0, len(getRank[0])):
    lastParetor.append(getRank[0][i][1])



xvals0, yvals0 = zip(*lastParetor)
plt.plot(xvals0, yvals0, 'o',  color = 'red')
plt.show()

