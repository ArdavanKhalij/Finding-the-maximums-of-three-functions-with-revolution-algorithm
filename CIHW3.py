##################################################### Libraries #######################################################
import math
import random
from numpy.random import choice
import numpy as np
import copy
import statistics
from matplotlib import pyplot as plt
import time
##################################################### Libraries #######################################################
############################################### Sum of first n indexes ################################################
def NSum(list, n):
    sum=0
    for i in range(0, n+1):
        sum = sum + list[i]
    return sum
############################################### Sum of first n indexes ################################################
################################################### First function ####################################################
def FirstFunction(n, x):
    result = 10*n
    for i in range(0, n):
        result = result + (x[i]*x[i])-10*math.cos(2*3.14159265359*x[i])
    return result
################################################### First function ####################################################
################################################## Second function ####################################################
def SecondFunction(a, b, c, d, x):
    sigma1 = 0
    sigma2 = 0
    for i in range(0, d):
        sigma1 = sigma1 + (x[i]*x[i])
        sigma2 = sigma2 + math.cos(c*x[i])
    result = (-1*a*math.exp(-1*b*math.sqrt((1/d)*sigma1))) - (math.exp((1/d)*sigma2)) + a + math.exp(1)
    return result
################################################## Second function ####################################################
################################################### Third function ####################################################
def ThirdFunction(n, x):
    sigma = 0
    for i in range(0, n):
        if(x[i]>=-5.12 or x[i]<=5.12):
            sigma = sigma + (x[i]*x[i]) - 10*math.cos(2*3.14159265359*x[i])
        else:
            sigma = sigma + (10*x[i]*x[i])
    result = 10*n + sigma
    return result
################################################### Third function ####################################################
###################################################### Get input ######################################################
with open('input.txt', 'r') as f:
    f.readline()
    n1 = int(f.readline())
    f.readline()
    a = float(f.readline())
    f.readline()
    b =float(f.readline())
    f.readline()
    c = float(f.readline())
    f.readline()
    d = int(f.readline())
    f.readline()
    n3 = int(f.readline())
    f.readline()
    numberOfGenerations = int(f.readline())
    f.readline()
    StartFrom = float(f.readline())
    f.readline()
    Until = float(f.readline())
    f.readline()
    numberOfPopulation = int(f.readline())
    f.readline()
    tournomentPopulation = int(f.readline())
    f.readline()
    MatingPercentage = int(f.readline())
    f.readline()
    COf5Rule = float(f.readline())
    f.readline()
    Sigma = float(f.readline())
    f.readline()
    SigmaForSelfCorrection = float(f.readline())
f.close()
###################################################### Get input ######################################################
############################## Make the random primary population for the first function ##############################
PrimaryPopulation1 = []
for i in range(0, numberOfPopulation):
    PrimaryPopulation1.append([])
    for j in range(0, n1):
        PrimaryPopulation1[i].append(random.uniform(StartFrom, Until))
############################## Make the random primary population for the first function ##############################
############################# Make the random primary population for the second function ##############################
PrimaryPopulation2 = []
for i in range(0, numberOfPopulation):
    PrimaryPopulation2.append([])
    for j in range(0, d):
        PrimaryPopulation2[i].append(random.uniform(StartFrom, Until))
############################# Make the random primary population for the second function ##############################
############################## Make the random primary population for the third function ##############################
PrimaryPopulation3 = []
for i in range(0, numberOfPopulation):
    PrimaryPopulation3.append([])
    for j in range(0, n3):
        PrimaryPopulation3[i].append(random.uniform(StartFrom, Until))
############################## Make the random primary population for the third function ##############################
################################ Make the random primary Sigma for the first function #################################
SigmaPopulation1 = []
for i in range(0, numberOfPopulation):
    SigmaPopulation1.append([])
    for j in range(0, n1):
        SigmaPopulation1[i].append(random.gauss(0, SigmaForSelfCorrection))
################################ Make the random primary Sigma for the first function #################################
############################### Make the random primary Sigma for the second function #################################
SigmaPopulation2 = []
for i in range(0, numberOfPopulation):
    SigmaPopulation2.append([])
    for j in range(0, d):
        SigmaPopulation2[i].append(random.gauss(0, SigmaForSelfCorrection))
############################### Make the random primary Sigma for the second function #################################
################################ Make the random primary Sigma for the third function #################################
SigmaPopulation3 = []
for i in range(0, numberOfPopulation):
    SigmaPopulation3.append([])
    for j in range(0, n3):
        SigmaPopulation3[i].append(random.gauss(0, SigmaForSelfCorrection))
################################ Make the random primary Sigma for the third function #################################
############################################### Fitness of fist function ##############################################
def Fitness1(population):
    fitnessArray = []
    for i in range(0, len(population)):
        fitnessArray.append(10/(FirstFunction(n1, population[i])+1))
    return fitnessArray
############################################### Fitness of fist function ##############################################
############################################## Fitness of second function #############################################
def Fitness2(population):
    fitnessArray = []
    for i in range(0, len(population)):
        fitnessArray.append(10/(SecondFunction(a, b, c, d, population[i])+1))
    return fitnessArray
############################################## Fitness of second function #############################################
############################################## Fitness of third function ##############################################
def Fitness3(population):
    fitnessArray = []
    for i in range(0, len(population)):
        if ThirdFunction(n3, population[i]) >= 0:
            fitnessArray.append(10/(ThirdFunction(n3, population[i])+1))
    return fitnessArray
############################################## Fitness of third function ##############################################
########################################## Roulette wheel for first function ##########################################
def RouletteWheel1(population):
    sum = 0
    poss = []
    values = Fitness1(population)
    for i in range(0, len(values)):
        sum = sum + values[i]
    for i in range(0, len(values)):
        poss.append((values[i]) / sum)
    x1 = 0.0
    for i in range(0, len(poss) - 1):
        x1 = x1 + poss[i]
    x2 = len(poss)
    poss[x2 - 1] = 1 - x1
    q = list(range(0, len(population)))
    parents = []
    for i in range(0, len(population)):
        draw = choice(q, 1, p=poss)
        parents.append(population[draw[0]])
    return parents
########################################## Roulette wheel for first function ##########################################
######################################### Roulette wheel for second function ##########################################
def RouletteWheel2(population):
    sum = 0
    poss = []
    values = Fitness2(population)
    for i in range(0, len(values)):
        sum = sum + values[i]
    for i in range(0, len(values)):
        poss.append((values[i]) / sum)
    x1 = 0.0
    for i in range(0, len(poss) - 1):
        x1 = x1 + poss[i]
    x2 = len(poss)
    poss[x2 - 1] = 1 - x1
    q = list(range(0, len(population)))
    parents = []
    for i in range(0, len(population)):
        draw = choice(q, 1, p=poss)
        parents.append(population[draw[0]])
    return parents
######################################### Roulette wheel for second function ##########################################
########################################## Roulette wheel for third function ##########################################
def RouletteWheel3(population):
    sum = 0
    poss = []
    values = Fitness3(population)
    for i in range(0, len(values)):
        sum = sum + values[i]
    for i in range(0, len(values)):
        poss.append((values[i]) / sum)
    x1 = 0.0
    for i in range(0, len(poss) - 1):
        x1 = x1 + poss[i]
    x2 = len(poss)
    poss[x2 - 1] = 1 - x1
    q = list(range(0, len(population)))
    parents = []
    for i in range(0, len(population)):
        draw = choice(q, 1, p=poss)
        parents.append(population[draw[0]])
    return parents
########################################## Roulette wheel for third function ##########################################
################################# Stochastic universal sampling for the first function ################################
def SUS1(population):
    n = len(population)
    sum = 0
    startPoint = random.uniform(0, 1/n)
    choosingVector = np.arange(startPoint, 1.0, 1/n)
    poss = []
    values = Fitness1(population)
    for i in range(0, len(values)):
        sum = sum + values[i]
    for i in range(0, len(values)):
        poss.append((values[i]) / sum)
    x1 = 0.0
    for i in range(0, len(poss) - 1):
        x1 = x1 + poss[i]
    x2 = len(poss)
    poss[x2 - 1] = 1 - x1
    q = []
    for i in range(0, len(population)):
        for j in range(0, len(poss)):
            if(choosingVector[i]<NSum(poss, j)):
                q.append(population[j])
                break
    return q
################################ Stochastic universal sampling for the first function #################################
################################ Stochastic universal sampling for the second function ################################
def SUS2(population):
    n = len(population)
    sum = 0
    startPoint = random.uniform(0, 1 / n)
    choosingVector = np.arange(startPoint, 1.0, 1 / n)
    poss = []
    values = Fitness2(population)
    for i in range(0, len(values)):
        sum = sum + values[i]
    for i in range(0, len(values)):
        poss.append((values[i]) / sum)
    x1 = 0.0
    for i in range(0, len(poss) - 1):
        x1 = x1 + poss[i]
    x2 = len(poss)
    poss[x2 - 1] = 1 - x1
    q = []
    for i in range(0, len(population)):
        for j in range(0, len(poss)):
            if (choosingVector[i] < NSum(poss, j)):
                q.append(population[j])
                break
    return q
################################ Stochastic universal sampling for the second function ################################
################################ Stochastic universal sampling for the third function #################################
def SUS3(population):
    n = len(population)
    sum = 0
    startPoint = random.uniform(0, 1 / n)
    choosingVector = np.arange(startPoint, 1.0, 1 / n)
    poss = []
    values = Fitness3(population)
    for i in range(0, len(values)):
        sum = sum + values[i]
    for i in range(0, len(values)):
        poss.append((values[i]) / sum)
    x1 = 0.0
    for i in range(0, len(poss) - 1):
        x1 = x1 + poss[i]
    x2 = len(poss)
    poss[x2 - 1] = 1 - x1
    q = []
    for i in range(0, len(population)):
        for j in range(0, len(poss)):
            if (choosingVector[i] < NSum(poss, j)):
                q.append(population[j])
                break
    return q
################################ Stochastic universal sampling for the third function #################################
################################## Tournoment sampling method for the first function ##################################
def TournomentSM1(population):
    result = []
    for i in range(0, len(population)):
        draw = random.choices(population, k=tournomentPopulation)
        k = Fitness1(draw)
        k2 = max(k)
        result.append(draw[k.index(k2)])
    return result
################################## Tournoment sampling method for the first function ##################################
################################# Tournoment sampling method for the second function ##################################
def TournomentSM2(population):
    result = []
    for i in range(0, len(population)):
        draw = random.choices(population, k=tournomentPopulation)
        k = Fitness2(draw)
        k2 = max(k)
        result.append(draw[k.index(k2)])
    return result
################################# Tournoment sampling method for the second function ##################################
################################## Tournoment sampling method for the third function ##################################
def TournomentSM3(population):
    result = []
    for i in range(0, len(population)):
        draw = random.choices(population, k=tournomentPopulation)
        k = Fitness3(draw)
        k2 = max(k)
        result.append(draw[k.index(k2)])
    return result
################################## Tournoment sampling method for the third function ##################################
######################################################## Mating #######################################################
def Mating(population):
    i = 0
    children = []
    while(i<len(population)):
        percent = np.random.uniform(0.0, 100.0)
        if(percent<=MatingPercentage):
            k = random.randint(0, len(population[i])-1)
            for j in range(k, len(population[i])):
                population[i][j] = (population[i][j] + population[i+1][j])/2
                population[i + 1][j] = (population[i][j] + population[i + 1][j]) / 2
            children.append(population[i])
            children.append(population[i+1])
        else:
            children.append(population[i])
            children.append(population[i+1])
        i = i + 2
    return children
######################################################## Mating #######################################################
#################################################### Sigma changer ####################################################
def SigmaChanger(counter11, counter21):
    result = counter11 / counter21
    global Sigma
    if result > 1/5:
        if ((Sigma / COf5Rule) > 2*StartFrom) and ((Sigma / COf5Rule) < 2*Until):
            Sigma = Sigma / COf5Rule
    elif result < 1/5:
        if ((Sigma * COf5Rule) > 2*StartFrom) and ((Sigma * COf5Rule) < 2*Until):
            Sigma = Sigma * COf5Rule
    else:
        Sigma = Sigma
#################################################### Sigma changer ####################################################
############################################ Mutation 1/5 for first function ##########################################
def OneFivthSigmaMutation1(population):
    MP = copy.deepcopy(population)
    global Sigma
    counter11 = 0
    counter21 = 0
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            k = random.gauss(0, Sigma)
            if ((MP[i][j] + k) <= Until) and ((MP[i][j] + k) >= StartFrom):
                MP[i][j] = MP[i][j] + k
        if (Fitness1([MP[i]]))[0]>(Fitness1([population[i]]))[0]:
            counter11 = counter11 + 1
        else:
            for h in range(0, n1):
                MP[i][h] = population[i][h]
        counter21 = counter21 + 1
    SigmaChanger(counter11, counter21)
    return MP
############################################ Mutation 1/5 for first function ##########################################
########################################### Mutation 1/5 for second function ##########################################
def OneFivthSigmaMutation2(population):
    MP = copy.deepcopy(population)
    global Sigma
    counter12 = 0
    counter22 = 0
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            k = random.gauss(0, Sigma)
            if ((MP[i][j] + k) <= Until) and ((MP[i][j] + k) >= StartFrom):
                MP[i][j] = MP[i][j] + k
        if (Fitness2([MP[i]]))[0] > (Fitness2([population[i]]))[0]:
            counter12 = counter12 + 1
        else:
            for h in range(0, d):
                MP[i][h] = population[i][h]
        counter22 = counter22 + 1
    SigmaChanger(counter12, counter22)
    return MP
########################################### Mutation 1/5 for second function ##########################################
############################################ Mutation 1/5 for third function ##########################################
def OneFivthSigmaMutation3(population):
    MP = copy.deepcopy(population)
    global Sigma
    counter13 = 0
    counter23 = 0
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            k = random.gauss(0, Sigma)
            if ((MP[i][j] + k) <= Until) and ((MP[i][j] + k) >= StartFrom):
                MP[i][j] = MP[i][j] + k
        if (Fitness3([MP[i]]))[0] > (Fitness3([population[i]]))[0]:
            counter13 = counter13 + 1
        else:
            for h in range(0, n3):
                MP[i][h] = population[i][h]
        counter23 = counter23 + 1
    SigmaChanger(counter13, counter23)
    return MP
############################################ Mutation 1/5 for third function ##########################################
#################################### Self correction mutation for the first function ##################################
def SelfCorrectionMutation1(population):
    global SigmaPopulation1
    SigmaPopulation = OneFivthSigmaMutation1(SigmaPopulation1)
    SigmaPopulation1 = copy.deepcopy(SigmaPopulation)
    MP = copy.deepcopy(population)
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            MP[i][j] = SigmaPopulation1[i][j] + population[i][j]
    return MP
#################################### Self correction mutation for the first function ##################################
################################### Self correction mutation for the second function ##################################
def SelfCorrectionMutation2(population):
    global SigmaPopulation2
    SigmaPopulation = OneFivthSigmaMutation2(SigmaPopulation2)
    SigmaPopulation2 = copy.deepcopy(SigmaPopulation)
    MP = copy.deepcopy(population)
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            MP[i][j] = SigmaPopulation2[i][j] + population[i][j]
    return MP
################################### Self correction mutation for the second function ##################################
#################################### Self correction mutation for the third function ##################################
def SelfCorrectionMutation3(population):
    global SigmaPopulation3
    SigmaPopulation = OneFivthSigmaMutation3(SigmaPopulation3)
    SigmaPopulation3 = copy.deepcopy(SigmaPopulation)
    MP = copy.deepcopy(population)
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            MP[i][j] = SigmaPopulation3[i][j] + population[i][j]
    return MP
#################################### Self correction mutation for the third function ##################################
####################################################### Main loop #####################################################
MAX1 = []
AVG1 = []
for i in range(0, numberOfGenerations):
    Parents11 = RouletteWheel3(PrimaryPopulation3)
    Children11 = Mating(Parents11)
    Muted = OneFivthSigmaMutation3(Children11)
    choosingNextGeneration = RouletteWheel3(Muted)
    PrimaryPopulation3 = copy.deepcopy(choosingNextGeneration)
    k = Fitness3(PrimaryPopulation3)
    kk = k.index((max(Fitness3(PrimaryPopulation3))))
    print(PrimaryPopulation3[kk])
    print(max(k))
    MAX1.append(max(k))
    AVG1.append(statistics.mean(k))
    print(Sigma)
    print("----------------------------------------")
####################################################### Main loop #####################################################
plt.plot(AVG1)
plt.plot(MAX1)
plt.show()
