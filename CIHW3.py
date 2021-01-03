##################################################### Libraries #######################################################
import math
import random
from numpy.random import choice
import numpy as np
import copy
import statistics
from matplotlib import pyplot as plt
import scipy.stats as ss
import time
from multiprocessing import *
import multiprocessing
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
start = time.time()
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
        SigmaPopulation1[i].append(random.uniform(0.001, SigmaForSelfCorrection))
################################ Make the random primary Sigma for the first function #################################
############################### Make the random primary Sigma for the second function #################################
SigmaPopulation2 = []
for i in range(0, numberOfPopulation):
    SigmaPopulation2.append([])
    for j in range(0, d):
        SigmaPopulation2[i].append(random.uniform(0.001, SigmaForSelfCorrection))
############################### Make the random primary Sigma for the second function #################################
################################ Make the random primary Sigma for the third function #################################
SigmaPopulation3 = []
for i in range(0, numberOfPopulation):
    SigmaPopulation3.append([])
    for j in range(0, n3):
        SigmaPopulation3[i].append(random.uniform(0.001, SigmaForSelfCorrection))
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
########################################## Direct Ranking for first function ##########################################
def DirectRanking1(population):
    MP = Fitness1(population)
    values = ss.rankdata(MP)
    poss = []
    sum = 0
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
########################################## Direct Ranking for first function ##########################################
######################################### Direct Ranking for second function ##########################################
def DirectRanking2(population):
    MP = Fitness2(population)
    values = ss.rankdata(MP)
    poss = []
    sum = 0
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
######################################### Direct Ranking for second function ##########################################
########################################## Direct Ranking for third function ###########################################
def DirectRanking3(population):
    MP = Fitness3(population)
    values = ss.rankdata(MP)
    poss = []
    sum = 0
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
########################################## Direct Ranking for third function ##########################################
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
################################################## Mutation for Sigma #################################################
def MFSCM1(population):
    MP = copy.deepcopy(population)
    tuPrime = 1/(math.sqrt(2*len(population[0])))
    tu = 1/(math.sqrt(2*(math.sqrt(len(population[0])))))
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            k1 = random.gauss(0, 1)
            k2 = random.gauss(0, 1)
            SP = population[i][j]*math.exp(tuPrime*k1+tu*k2)
            if ((SP<=Until)and(SP>=StartFrom)):
                MP[i][j] = SP
    return MP
def MFSCM2(population):
    MP = copy.deepcopy(population)
    # tuPrime = 1/(math.sqrt(2*len(population[0])))
    tu = 1/(math.sqrt(2*(math.sqrt(len(population[0])))))
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            k1 = random.gauss(0, 1)
            # k2 = random.gauss(0, 1)
            SP = population[i][j]*math.exp(tu*k1)
            if ((SP<=Until)and(SP>=StartFrom)):
                MP[i][j] = SP
    return MP
################################################## Mutation for Sigma #################################################
#################################### Self correction mutation for the first function ##################################
def SelfCorrectionMutation1(population, SigmaPopulation1):
    SigmaPopulation = MFSCM2(SigmaPopulation1)
    SigmaPopulation1 = copy.deepcopy(SigmaPopulation)
    MP = copy.deepcopy(population)
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            k = population[i][j] + (SigmaPopulation1[i][j] * random.gauss(0,1))
            if ((k>=StartFrom) and (k<=Until)):
                MP[i][j] = k
        if Fitness1([MP[i]])[0]<=Fitness1([population[i]])[0]:
            MP[i] = copy.deepcopy(population[i])
    return MP
#################################### Self correction mutation for the first function ##################################
################################### Self correction mutation for the second function ##################################
def SelfCorrectionMutation2(population, SigmaPopulation2):
    SigmaPopulation = MFSCM2(SigmaPopulation2)
    SigmaPopulation2 = copy.deepcopy(SigmaPopulation)
    MP = copy.deepcopy(population)
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            k = population[i][j] + (SigmaPopulation2[i][j] * random.gauss(0, 1))
            if ((k >= StartFrom) and (k <= Until)):
                MP[i][j] = k
        if Fitness2([MP[i]])[0]<=Fitness2([population[i]])[0]:
            MP[i] = copy.deepcopy(population[i])
    return MP
################################### Self correction mutation for the second function ##################################
#################################### Self correction mutation for the third function ##################################
def SelfCorrectionMutation3(population, SigmaPopulation3):
    SigmaPopulation = MFSCM2(SigmaPopulation3)
    SigmaPopulation3 = copy.deepcopy(SigmaPopulation)
    MP = copy.deepcopy(population)
    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            k = population[i][j] + (SigmaPopulation3[i][j] * random.gauss(0, 1))
            if ((k >= StartFrom) and (k <= Until)):
                MP[i][j] = k
        if Fitness2([MP[i]])[0]<=Fitness3([population[i]])[0]:
            MP[i] = copy.deepcopy(population[i])
    return MP
#################################### Self correction mutation for the third function ##################################
####################################################### Main loop #####################################################

MAX1 = []
AVG1 = []
MAX2 = []
AVG2 = []
MAX3 = []
AVG3 = []
MAX4 = []
AVG4 = []
MAX5 = []
AVG5 = []
MAX6 = []
AVG6 = []
MAX7 = []
AVG7 = []
MAX8 = []
AVG8 = []
MAX9 = []
AVG9 = []
MAX10 = []
AVG10 = []
MAX11 = []
AVG11 = []
MAX12 = []
AVG12 = []
MAX13 = []
AVG13 = []
MAX14 = []
AVG14 = []
MAX15 = []
AVG15 = []
MAX16 = []
AVG16 = []
MAX17 = []
AVG17 = []
MAX18 = []
AVG18 = []
PrimaryPopulation11=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation12=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation13=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation14=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation15=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation16=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation17=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation18=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation19=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation110=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation111=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation112=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation113=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation114=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation115=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation116=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation117=copy.deepcopy(PrimaryPopulation1)
PrimaryPopulation118=copy.deepcopy(PrimaryPopulation1)
SigmaPopulation11=copy.deepcopy(SigmaPopulation1)
SigmaPopulation12=copy.deepcopy(SigmaPopulation1)
SigmaPopulation13=copy.deepcopy(SigmaPopulation1)
SigmaPopulation14=copy.deepcopy(SigmaPopulation1)
SigmaPopulation15=copy.deepcopy(SigmaPopulation1)
SigmaPopulation16=copy.deepcopy(SigmaPopulation1)
SigmaPopulation17=copy.deepcopy(SigmaPopulation1)
SigmaPopulation18=copy.deepcopy(SigmaPopulation1)
SigmaPopulation19=copy.deepcopy(SigmaPopulation1)
SigmaPopulation110=copy.deepcopy(SigmaPopulation1)
SigmaPopulation111=copy.deepcopy(SigmaPopulation1)
SigmaPopulation112=copy.deepcopy(SigmaPopulation1)
SigmaPopulation113=copy.deepcopy(SigmaPopulation1)
SigmaPopulation114=copy.deepcopy(SigmaPopulation1)
SigmaPopulation115=copy.deepcopy(SigmaPopulation1)
SigmaPopulation116=copy.deepcopy(SigmaPopulation1)
SigmaPopulation117=copy.deepcopy(SigmaPopulation1)
SigmaPopulation118=copy.deepcopy(SigmaPopulation1)

for i in range(0, numberOfGenerations):
    Parents11 = RouletteWheel1(PrimaryPopulation11)
    Children11 = Mating(Parents11)
    Muted11 = SelfCorrectionMutation1(Children11, SigmaPopulation11)
    choosingNextGeneration11 = RouletteWheel1(Muted11)
    PrimaryPopulation11 = copy.deepcopy(choosingNextGeneration11)
    k = Fitness1(PrimaryPopulation11)
    kk = k.index((max(Fitness1(PrimaryPopulation11))))
    print(PrimaryPopulation11[kk])
    print(max(k))
    MAX1.append(max(k))
    AVG1.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents12 = SUS1(PrimaryPopulation12)
    Children12 = Mating(Parents12)
    Muted12 = SelfCorrectionMutation1(Children12, SigmaPopulation12)
    choosingNextGeneration12 = SUS1(Muted12)
    PrimaryPopulation12 = copy.deepcopy(choosingNextGeneration12)
    k = Fitness1(PrimaryPopulation12)
    kk = k.index((max(Fitness1(PrimaryPopulation12))))
    print(PrimaryPopulation12[kk])
    print(max(k))
    MAX2.append(max(k))
    AVG2.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents13 = TournomentSM1(PrimaryPopulation13)
    Children13 = Mating(Parents13)
    Muted13 = SelfCorrectionMutation1(Children13, SigmaPopulation13)
    choosingNextGeneration13 = TournomentSM1(Muted13)
    PrimaryPopulation13 = copy.deepcopy(choosingNextGeneration13)
    k = Fitness1(PrimaryPopulation13)
    kk = k.index((max(Fitness1(PrimaryPopulation13))))
    print(PrimaryPopulation13[kk])
    print(max(k))
    MAX3.append(max(k))
    AVG3.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents14 = TournomentSM1(PrimaryPopulation14)
    Children14 = Mating(Parents14)
    Muted14 = SelfCorrectionMutation1(Children14, SigmaPopulation14)
    choosingNextGeneration14 = SUS1(Muted14)
    PrimaryPopulation14 = copy.deepcopy(choosingNextGeneration14)
    k = Fitness1(PrimaryPopulation14)
    kk = k.index((max(Fitness1(PrimaryPopulation14))))
    print(PrimaryPopulation14[kk])
    print(max(k))
    MAX4.append(max(k))
    AVG4.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents15 = TournomentSM1(PrimaryPopulation15)
    Children15 = Mating(Parents15)
    Muted15 = SelfCorrectionMutation1(Children15, SigmaPopulation15)
    choosingNextGeneration15 = RouletteWheel1(Muted15)
    PrimaryPopulation15 = copy.deepcopy(choosingNextGeneration15)
    k = Fitness1(PrimaryPopulation15)
    kk = k.index((max(Fitness1(PrimaryPopulation15))))
    print(PrimaryPopulation15[kk])
    print(max(k))
    MAX5.append(max(k))
    AVG5.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents16 = RouletteWheel1(PrimaryPopulation16)
    Children16 = Mating(Parents16)
    Muted16 = SelfCorrectionMutation1(Children16, SigmaPopulation16)
    choosingNextGeneration16 = SUS1(Muted16)
    PrimaryPopulation16 = copy.deepcopy(choosingNextGeneration16)
    k = Fitness1(PrimaryPopulation16)
    kk = k.index((max(Fitness1(PrimaryPopulation16))))
    print(PrimaryPopulation16[kk])
    print(max(k))
    MAX6.append(max(k))
    AVG6.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents17 = RouletteWheel1(PrimaryPopulation17)
    Children17 = Mating(Parents17)
    Muted17 = SelfCorrectionMutation1(Children17, SigmaPopulation17)
    choosingNextGeneration17 = TournomentSM1(Muted17)
    PrimaryPopulation17 = copy.deepcopy(choosingNextGeneration17)
    k = Fitness1(PrimaryPopulation17)
    kk = k.index((max(Fitness1(PrimaryPopulation17))))
    print(PrimaryPopulation17[kk])
    print(max(k))
    MAX7.append(max(k))
    AVG7.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents18 = SUS1(PrimaryPopulation18)
    Children18 = Mating(Parents18)
    Muted18 = SelfCorrectionMutation1(Children18, SigmaPopulation18)
    choosingNextGeneration18 = RouletteWheel1(Muted18)
    PrimaryPopulation18 = copy.deepcopy(choosingNextGeneration18)
    k = Fitness1(PrimaryPopulation18)
    kk = k.index((max(Fitness1(PrimaryPopulation18))))
    print(PrimaryPopulation18[kk])
    print(max(k))
    MAX8.append(max(k))
    AVG8.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents19 = SUS1(PrimaryPopulation19)
    Children19 = Mating(Parents19)
    Muted19 = SelfCorrectionMutation1(Children19, SigmaPopulation19)
    choosingNextGeneration19 = TournomentSM1(Muted19)
    PrimaryPopulation19 = copy.deepcopy(choosingNextGeneration19)
    k = Fitness1(PrimaryPopulation19)
    kk = k.index((max(Fitness1(PrimaryPopulation19))))
    print(PrimaryPopulation19[kk])
    print(max(k))
    MAX9.append(max(k))
    AVG9.append(statistics.mean(k))
    print("----------------------------------------")
    ####################
    Parents110 = RouletteWheel1(PrimaryPopulation110)
    Children110 = Mating(Parents110)
    Muted110 = OneFivthSigmaMutation1(Children110)
    choosingNextGeneration110 = RouletteWheel1(Muted110)
    PrimaryPopulation110 = copy.deepcopy(choosingNextGeneration110)
    k = Fitness1(PrimaryPopulation110)
    kk = k.index((max(Fitness1(PrimaryPopulation110))))
    print(PrimaryPopulation110[kk])
    print(max(k))
    MAX10.append(max(k))
    AVG10.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents111 = SUS1(PrimaryPopulation111)
    Children111 = Mating(Parents111)
    Muted111 = OneFivthSigmaMutation1(Children111)
    choosingNextGeneration111 = SUS1(Muted111)
    PrimaryPopulation111 = copy.deepcopy(choosingNextGeneration111)
    k = Fitness1(PrimaryPopulation111)
    kk = k.index((max(Fitness1(PrimaryPopulation111))))
    print(PrimaryPopulation111[kk])
    print(max(k))
    MAX11.append(max(k))
    AVG11.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents112 = TournomentSM1(PrimaryPopulation112)
    Children112 = Mating(Parents112)
    Muted112 = OneFivthSigmaMutation1(Children112)
    choosingNextGeneration112 = TournomentSM1(Muted112)
    PrimaryPopulation112 = copy.deepcopy(choosingNextGeneration112)
    k = Fitness1(PrimaryPopulation112)
    kk = k.index((max(Fitness1(PrimaryPopulation112))))
    print(PrimaryPopulation112[kk])
    print(max(k))
    MAX12.append(max(k))
    AVG12.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents113 = TournomentSM1(PrimaryPopulation113)
    Children113 = Mating(Parents113)
    Muted113 = OneFivthSigmaMutation1(Children113)
    choosingNextGeneration113 = SUS1(Muted113)
    PrimaryPopulation113 = copy.deepcopy(choosingNextGeneration113)
    k = Fitness1(PrimaryPopulation113)
    kk = k.index((max(Fitness1(PrimaryPopulation113))))
    print(PrimaryPopulation113[kk])
    print(max(k))
    MAX13.append(max(k))
    AVG13.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents114 = TournomentSM1(PrimaryPopulation114)
    Children114 = Mating(Parents114)
    Muted114 = OneFivthSigmaMutation1(Children114)
    choosingNextGeneration114 = RouletteWheel1(Muted114)
    PrimaryPopulation114 = copy.deepcopy(choosingNextGeneration114)
    k = Fitness1(PrimaryPopulation114)
    kk = k.index((max(Fitness1(PrimaryPopulation114))))
    print(PrimaryPopulation114[kk])
    print(max(k))
    MAX14.append(max(k))
    AVG14.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents115 = RouletteWheel1(PrimaryPopulation115)
    Children115 = Mating(Parents115)
    Muted115 = OneFivthSigmaMutation1(Children115)
    choosingNextGeneration115 = SUS1(Muted115)
    PrimaryPopulation115 = copy.deepcopy(choosingNextGeneration115)
    k = Fitness1(PrimaryPopulation115)
    kk = k.index((max(Fitness1(PrimaryPopulation115))))
    print(PrimaryPopulation115[kk])
    print(max(k))
    MAX15.append(max(k))
    AVG15.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents116 = RouletteWheel1(PrimaryPopulation116)
    Children116 = Mating(Parents116)
    Muted116 = OneFivthSigmaMutation1(Children116)
    choosingNextGeneration116 = TournomentSM1(Muted116)
    PrimaryPopulation116 = copy.deepcopy(choosingNextGeneration116)
    k = Fitness1(PrimaryPopulation116)
    kk = k.index((max(Fitness1(PrimaryPopulation116))))
    print(PrimaryPopulation116[kk])
    print(max(k))
    MAX16.append(max(k))
    AVG16.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents117 = SUS1(PrimaryPopulation117)
    Children117 = Mating(Parents117)
    Muted117 = OneFivthSigmaMutation1(Children117)
    choosingNextGeneration117 = RouletteWheel1(Muted117)
    PrimaryPopulation117 = copy.deepcopy(choosingNextGeneration117)
    k = Fitness1(PrimaryPopulation117)
    kk = k.index((max(Fitness1(PrimaryPopulation117))))
    print(PrimaryPopulation117[kk])
    print(max(k))
    MAX17.append(max(k))
    AVG17.append(statistics.mean(k))
    print("----------------------------------------")
    ###################
    Parents118 = SUS1(PrimaryPopulation118)
    Children118 = Mating(Parents118)
    Muted118 = OneFivthSigmaMutation1(Children118)
    choosingNextGeneration118 = TournomentSM1(Muted118)
    PrimaryPopulation118 = copy.deepcopy(choosingNextGeneration118)
    k = Fitness1(PrimaryPopulation118)
    kk = k.index((max(Fitness1(PrimaryPopulation118))))
    print(PrimaryPopulation118[kk])
    print(max(k))
    MAX18.append(max(k))
    AVG18.append(statistics.mean(k))
    print("----------------------------------------")
####################################################### Main loop #####################################################
def chartAndTime():
    global start
    global MAX1
    global MAX2
    global MAX3
    global MAX4
    global MAX5
    global MAX6
    global MAX7
    global MAX8
    global MAX9
    global MAX10
    global MAX11
    global MAX12
    global MAX13
    global MAX14
    global MAX15
    global MAX16
    global MAX17
    global MAX18
    TIME = time.time()-start
    print(TIME)

    fig, axs = plt.subplots(3, 3)
    axs[0, 0].plot(MAX1, label='Self Correction')
    axs[0, 0].set_title('function1/RW/RW')
    axs[0, 1].plot(MAX2, 'tab:orange', label='Self Correction')
    axs[0, 1].set_title('function1/SUS/SUS')
    axs[0, 2].plot(MAX3, 'tab:green', label='Self Correction')
    axs[0, 2].set_title('function1/T/T')
    axs[1, 0].plot(MAX4, 'tab:red', label='Self Correction')
    axs[1, 0].set_title('function1/T/SUS')
    axs[1, 1].plot(MAX5, 'tab:blue', label='Self Correction')
    axs[1, 1].set_title('function1/T/RW')
    axs[1, 2].plot(MAX6, 'tab:olive', label='Self Correction')
    axs[1, 2].set_title('function1/RW/SUS')
    axs[2, 0].plot(MAX7, 'tab:pink', label='Self Correction')
    axs[2, 0].set_title('function1/RW/T')
    axs[2, 1].plot(MAX8, 'tab:brown', label='Self Correction')
    axs[2, 1].set_title('function1/SUS/RW')
    axs[2, 2].plot(MAX9, 'tab:purple', label='Self Correction')
    axs[2, 2].set_title('function1/SUS/T')

    axs[0, 0].plot(MAX10, 'tab:cyan', label='1/5 Rule')
    axs[0, 1].plot(MAX11, 'tab:cyan', label='1/5 Rule')
    axs[0, 2].plot(MAX12, 'tab:cyan', label='1/5 Rule')
    axs[1, 0].plot(MAX13, 'tab:cyan', label='1/5 Rule')
    axs[1, 1].plot(MAX14, 'tab:cyan', label='1/5 Rule')
    axs[1, 2].plot(MAX15, 'tab:cyan', label='1/5 Rule')
    axs[2, 0].plot(MAX16, 'tab:cyan', label='1/5 Rule')
    axs[2, 1].plot(MAX17, 'tab:cyan', label='1/5 Rule')
    axs[2, 2].plot(MAX18, 'tab:cyan', label='1/5 Rule')

    for ax in axs.flat:
        ax.set(xlabel='Number of generations', ylabel='Fitness from 10')

    for ax in axs.flat:
        ax.label_outer()

chartAndTime()
plt.show()
