import numpy as np
import random
import matplotlib.pyplot as plt

#init par

INF = -99999999


def dominate(p1, p2):
    if (p1[1][0] <= p2[1][0] and p1[1][1] <= p2[1][1]):
        if (p1[1][0] < p2[1][0] or p1[1][1] < p2[1][1]):
            return 1
        
    else:
        return -1
    return 0


# m =  2
# N : so mau vector 
# T : so vector lan can
# 

n = 2 # kich thuoc cua mot mau vector
m = 2 # so bai toan con
N = 100 # so cac mau vector
T = 10 # so vector lan can
EP = [] # tap dau ra 
lstVec = [] #tap cac vecor
generations = 300 # so cac the he
z = [INF, INF] # vector tham chieu, khoi tao bang vo cung
crossover_rate = 0.9
n_c = 6



def FitnessForChild(p):
    Msum = 0
    for i in range(1, n):
        Msum += p[i]

    g = 1 + 9 * (Msum) / (n - 1)
    f1 = p[0]
    f2 = g * (1 - np.sqrt(p[0] / g))
    return [f1, f2]


def CalculateDistance(a, b):
    point1 = np.array(a)
    point2 = np.array(b)
    return np.linalg.norm(point1 - point2)



# init vector


# For 2 object

def generate_uniform_2d_vector():
    vectors = []
    x = np.arange(0, 1, 1 / N)
    y = 1 - x
    for i in range(0, len(x)):
        vectors.append([x[i], y[i]])

    return vectors


lstVec = generate_uniform_2d_vector()


# init population

population = []

for i in range(0, N):
    coor = []
    for j in range(0, n):
        coor.append(random.random())

    fit = FitnessForChild(coor)
    population.append([coor, fit])



#so chieu chinh la so bai toan con -> m


#compute neighbor

lstNeighbor = []
for i in range(0, N):
    a = []
    for j in range(0, N):
        a.append([j, CalculateDistance(lstVec[i], lstVec[j])])

    a.sort(key = lambda coor : coor[1])
    b = []
    for j in range(0, T):
        b.append([a[j][0], lstVec[j]])

    lstNeighbor.append(b)

# lstNeighbor = [[1, 2, 3], [4, 5, 6]]

# nhu vay la ta se co duoc N bai toan
# Bay h ta di tinh toan cac vector

#update

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


        fit1 = FitnessForChild(pop1)
        fit2 = FitnessForChild(pop2)
        child1.append(pop1)
        child1.append(fit1)
        child2.append(pop2)
        child2.append(fit2)
        newLstChild.append(child1)
        newLstChild.append(child2)
        
    return newLstChild
    
#dung de tinh ham ung voi vector thu index nao do
def caculateWeightedFunction(pop, vec):
    weightedMax = - INF
    for i in range(0, m):
        c = vec[i] * abs(pop[1][i] - z[i])
        if (c > weightedMax):
            weightedMax = c

    return weightedMax



for k in range(0, generations):
    for i in range(0, N):
        a = random.choice(lstNeighbor[i])
        b = random.choice(lstNeighbor[i])
        y = CrossOver(population[a[0]], population[b[0]])
        # calculate WeightedFunction
        # Heuristic to imporve y
        # Update of z
                

        # Update of Neighboring Solutions
        for j in range(0, len(y)):
            for i in range(0, m):
                if (z[i] < y[j][1][i]):
                    z[i] = y[j][1][i]
            for x in lstNeighbor[i]:
                if (caculateWeightedFunction(y[j], lstVec[x[0]]) <= caculateWeightedFunction(population[x[0]], lstVec[x[0]])):
                    population[x[0]] = y[j].copy()
            isEqual = False
            for x in EP:
                c = dominate(y[j], x)
                if (c == 1):
                    EP.remove(x)
                elif (c == 0):
                    isEqual = True
                    break
            
            if (isEqual == False):
                check = True
                for x in EP:
                    if (dominate(x, y[j]) == 1):
                        check = False
                        break

                if (check):
                    EP.append(y[j])

    print(str(k) + "  " + str(len(EP)))

lastParetor = []
for i in range(0, len(EP)):
    lastParetor.append(EP[i][1])


xvals0, yvals0 = zip(*lastParetor)
plt.plot(xvals0, yvals0, 'o',  color = 'red')
plt.show()
    
