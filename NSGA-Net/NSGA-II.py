import torch
from torch import nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import time




def DecodingToList(encoding):
    nodeEncoding = []
    index = 0
    k = 1
    while index < len(encoding) - 1:
        nodeEncoding.append(encoding[index : (index + k)])
        index += k
        k += 1
    nodeEncoding.append([encoding[-1]])
    return nodeEncoding

class Node(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Node, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CustomBlock(nn.Module):
    def __init__(self, encoding, in_channels, out_channels):
        self.encodingList = DecodingToList(encoding)
        super(CustomBlock, self).__init__()
        self.init_node = Node(in_channels, out_channels)
        self.block = nn.ModuleList([Node(out_channels, out_channels) 
                                    for _ in range(len(self.encodingList) + 1)])

    def forward(self, x):
        # nhận giá trị cho node 0 trước giá trị cuối
        # nếu bit cuối khác 0, ta có cấu trúc dạng ResNet
        # mặc định nếu các bit khác chưa có đầu vào, nhận của input node
        outputs = [self.init_node(x)]
        outputs.append(self.block[0](outputs[0]))
        for index, layer in enumerate(self.block[1:], start=1):
            node_inputs = [outputs[j] for j, val 
                           in enumerate(self.encodingList[index - 1]) if val == '1']
            if len(node_inputs) == 0:
                node_inputs.append(outputs[0])

            node_inputs = sum(node_inputs)
            node_output = layer(node_inputs)
            outputs.append(node_output)
        return outputs[-1]

class Individual_Model(nn.Module):
    def __init__(self, encoding):
        super(Individual_Model, self).__init__()
        sp = len(encoding) // 3
        self.b1 = CustomBlock(encoding[:sp], in_channels=3, out_channels=16)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b2 = CustomBlock(encoding[sp:(2*sp)], in_channels=16, out_channels=32)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b3 = CustomBlock(encoding[(2*sp):], in_channels=32, out_channels=64)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fl = nn.Flatten()
        self.fc = nn.Linear(64 * 4 * 4, 10)
        self.apply(self.init_weights)

    def forward(self, x):
        x = self.b1(x)
        x = self.p1(x)
        x = self.b2(x)
        x = self.p2(x)
        x = self.b3(x)
        x = self.p3(x)
        x = self.fl(x)
        return self.fc(x)
    
    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(layer.weight)

class NSGA_Net:
    def __init__(self, sizePop = 20, size = 10, k = 10
                 , n = 2, crossover_rate = 0.9, mutation_rate = 0.01, generation = 3):
        self.sizePop = sizePop
        self.size = size
        self.k = k
        self.n = n
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generation = generation
        self.population = []
        self.INF = 999999999
        self.device = torch.device('cuda:0')
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.dataset_size = len(self.trainset)
        self.indices = list(range(self.dataset_size))
        self.train_indices, val_indices = train_test_split(self.indices, test_size=0.2, random_state=42)
        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)
        self.trainloader = DataLoader(self.trainset, batch_size=64, sampler=self.train_sampler)
        self.valloader = DataLoader(self.trainset, batch_size=64, sampler=self.val_sampler)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=False)
        self.learning_rate = 0.1
        self.all = []

    def dominate(self, p1, p2):
        if (p1[1][0] <= p2[1][0] and p1[1][1] <= p2[1][1]):
            if (p1[1][0] < p2[1][0] or p1[1][1] < p2[1][1]):
                return True
        
        return False    

    def GeneratePopulation(self):
        for i in range(0, self.sizePop):
            print("Đang khởi tạo cá thể thứ {}".format(i))
            coor = []
            for j in range(0, self.n):
                p = random.random()
                if p >= 0.5:
                    coor.append(1)
                else:
                    coor.append(0)

            fit = self.Fitness(coor)
            self.population.append([coor, fit])
            print("Cá thể thứ {} có accuracy : {:.3f} %".format(i, 100 * (1 - fit[0])))

    def Fitness(self, individual):
        model = Individual_Model(individual)
        model.to(self.device)
        fitnessList = []
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        for epoch in range(5):
            for i, (images, labels) in enumerate(self.trainloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                ouputs = model(images)
                loss = criterion(ouputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        with torch.no_grad():
            correct = 0
            total = 0
            totalTime = 0
            for images, labels in self.valloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                startTime = time.time()
                ouputs = model(images)
                endTime = time.time()
                totalTime += (endTime - startTime)
                predicted = torch.max(ouputs, 1)[1] 
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            fitnessList.append(1 - (correct / total))          
            fitnessList.append(totalTime / 10000)
        return fitnessList
    
    def Mutate(self, p):
        newChild = p.copy()
        count = 0
        if (random.random() < self.mutation_rate):
            for i in range(self.n):
                if random.random() > 0.5 and count == 0:
                    newChild[0][i] = 1 - newChild[0][i]
                    count = 1
                    break

        fit = self.Fitness(newChild[0])
        newChild[1] = fit
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
        child = []
        isCross = random.random()
        if isCross <= self.crossover_rate:
            # crossover_point = np.random.randint(1, self.n - 1)
            # child1[crossover_point:], child2[crossover_point:] = p2[crossover_point:], p1[crossover_point:]
            # fit1 = self.Fitness(child1)
            # fit2 = self.Fitness(child2)
            # newLstChild.append([child1, fit1])
            # newLstChild.append([child2, fit2])
            for i in range(self.n):
                if p1[i] == p2[i]:
                    child.append(p1[i])
                else:
                    rand = random.random()
                    if rand >= 0.5:
                        child.append(1)
                    else:
                        child.append(0)

            fit = self.Fitness(child)
            newLstChild.append([child, fit])
                

    
        return newLstChild
        # newLstChild = []
        # child1 = p1.copy()
        # child2 = p2.copy()
        # isCross = random.random()
        # if (isCross <= self.crossover_rate):
        #     crossover_point = np.random.randint(1, self.n - 1)
        #     child1[crossover_point:], child2[crossover_point:] = p2[crossover_point:], p1[crossover_point:]
        #     fit1 = self.Fitness(child1)
        #     fit2 = self.Fitness(child2)
        #     newLstChild.append([child1, fit1])
        #     newLstChild.append([child2, fit2])
        
        # return newLstChild

    def BreedPopulation(self):
        child = self.population.copy()
        for i in range(0, self.k):
            a = random.choice(self.population)
            b = random.choice(self.population)
            c = self.CrossOver(a[0], b[0])
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
                    if (self.dominate(q, p)):
                        n1 += 1
                    elif (self.dominate(p, q)):
                        S.append(q)

                if (n1 == 0):
                    Rank0.append(p)

                listDominate[str(p)] = [S, n1]
            else :
                listDominate[str(p)][0] = 2 * listDominate[str(p)][0]
                listDominate[str(p)][1] = 2 * listDominate[str(p)][1]
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

    def CrowdingDistance(self, pop):
        l = len(pop)
        Ldistance = {}
        for i in range(0, l):
            Ldistance[str(pop[i])] = 0
        for i in range(0, 2):
            a = pop.copy()
            a.sort(key = lambda coor : coor[1][i])
            fmin = a[0][1][i]
            fmax = a[l - 1][1][i]
            Ldistance[str(a[0])] += self.INF
            Ldistance[str(a[l - 1])] += self.INF
            for j in range(1, l - 2):
                Ldistance[str(a[j])] += (a[j + 1][1][i] - a[j - 1][1][i])/(fmax - fmin)

        ans = []
        for j in range(0, l):
            newDis = []
            newDis.append(pop[j])
            newDis.append(Ldistance[str(pop[j])])
            ans.append(newDis)
        return ans

    def isEqualPop(self, p1, p2):
        for i in range(0, self.n):
            if (p1[0][i] != p2[0][i]):
                return False
            
        return True
    def SelectPopulation(self):
        F = self.FastNonDominatedSort()
        sizeToTake = self.size + len(F[0])
        if sizeToTake > len(self.population):
            sizeToTake = self.population
        nextGene = []
        i = 0
        while i < len(F) and (len(nextGene) + len(F[i])) <= sizeToTake:
            nextGene += F[i]
            i += 1
        if i < len(F):
            lstCrow = self.CrowdingDistance(F[i])
            lstCrow.sort(key = lambda coor : coor[1], reverse=True)
            l = len(nextGene)
            for j in range(0, self.size - l):
                nextGene.append(lstCrow[j][0])
       
        self.population = nextGene.copy()



    def Run(self):
        for i in range(self.generation):
            print("the he thu : " + str(i))
            self.fixPopulation()
            self.SelectPopulation()
            self.BreedPopulation()
            self.MutatePopulation()
            for i in range(len(self.population)):
                self.all.append(self.population[i])

    

    def fixPopulationForLastPareto(self):
        popDic = {}
        for i in range(0, len(self.all)):
            popDic[str(self.all[i])] = 0

        index = []
        for i in range(0, len(self.all)):
            if popDic[str(self.all[i])] == 0:
                popDic[str(self.all[i])] = 1
            else:
                index.append(i)

        newPop = []
        for i in range(0, len(self.all)):
            if i not in index:
                newPop.append(self.all[i])

        self.all = newPop.copy()

    def DrawAllIndividual(self):
        self.fixPopulationForLastPareto()
        Fit = []
        for p in self.all:
            Fit.append(p[1])

        Rank0 = []
        for p in self.all:
            isNonDominated = 1
            for q in self.all:
                if (self.dominate(q, p)):
                   isNonDominated = 0
                   break
            
            if (isNonDominated == 1):
                Rank0.append(p[1])

        xvals0, yvals0 = zip(*Fit)
        xvals1, yvals1 = zip(*Rank0)
        plt.plot(xvals0, yvals0, 'o',  color = 'red')
        plt.plot(xvals1, yvals1, 'o',  color = 'blue')
        plt.show()

    def DrawPareto(self):
        Rank0 = []
        for p in self.population:
            isNonDominated = 1
            for q in self.population:
                if (self.dominate(q, p)):
                   isNonDominated = 0
                   break
            
            if (isNonDominated == 1):
                Rank0.append(p[1])

        xvals0, yvals0 = zip(*Rank0)
        plt.plot(xvals0, yvals0, 'o',  color = 'red')
        plt.show()


#class B

newProblem = NSGA_Net(sizePop=20,
                      size=10,
                      k = 10,
                      n=33,
                      crossover_rate=0.9,
                      mutation_rate=0.1,
                      generation=10
                      )
newProblem.GeneratePopulation()
newProblem.Run()
newProblem.DrawPareto()

        





