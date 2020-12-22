##################################################### Libraries #######################################################
import math
import random
from numpy.random import choice
import numpy as np
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
    for i in range(1, n+1):
        print(i)
        result = result + (x[i]*x[i])-10*math.cos(2*3.14159265359*x[i])
    return result
################################################### First function ####################################################
################################################## Second function ####################################################
def SecondFunction(a, b, c, d, x):
    sigma1 = 0
    sigma2 = 0
    for i in range(1, d+1):
        sigma1 = sigma1 + (x[i]*x[i])
        sigma2 = sigma2 + math.cos(c*x[i])
    result = (-1*a*math.exp(-1*b*math.sqrt((1/d)*sigma1))) - (math.exp((1/d)*sigma2)) + a + math.exp(1)
    return result
################################################## Second function ####################################################
################################################### Third function ####################################################
def ThirdFunction(n, x):
    sigma = 0
    for i in range(0, n):
        if(x[i]>=-5.12 and x[i]<=5.12):
            sigma = sigma + (x[i]*x[i]) - 10*math.cos(2*3.14159265359*x[i])
        else:
            sigma = sigma + (10*x[i]*x[i])
    result = 10*n + sigma
    return result
################################################### Third function ####################################################
###################################################### Get input ######################################################
with open('input.txt', 'r') as f:
    n1 = int(f.readline())
    a = float(f.readline())
    b =float(f.readline())
    c = float(f.readline())
    d = int(f.readline())
    n3 = int(f.readline())
    numberOfGenerations = int(f.readline())
    StartFrom = float(f.readline())
    Until = float(f.readline())
    numberOfPopulation = int(f.readline())
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
############################################### Fitness of fist function ##############################################
def Fitness1(population):
    fitnessArray = []
    for i in range(0, len(population)):
        fitnessArray.append(FirstFunction(n1, population[i]))
    return fitnessArray
############################################### Fitness of fist function ##############################################
############################################## Fitness of second function #############################################
def Fitness2(population):
    fitnessArray = []
    for i in range(0, len(population)):
        fitnessArray.append(SecondFunction(a, b, c, d, population[i]))
    return fitnessArray
############################################## Fitness of second function #############################################
############################################## Fitness of third function ##############################################
def Fitness3(population):
    fitnessArray = []
    for i in range(0, len(population)):
        fitnessArray.append(ThirdFunction(n3, population[i]))
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
################################# Stochastic universal sampling for the third function#################################
