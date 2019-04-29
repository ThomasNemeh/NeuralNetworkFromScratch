# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 00:01:54 2019

@author: pingo
"""

from abc import ABC
import numpy as np

class Module(ABC):
    def __init__(self, numUnits, weightList, inputs, activations, beta, lam):
        self.weightMatrix = np.zeros([numUnits,numUnits])
        self.weightList = weightList
        self.beta = beta
        self.lam = lam
        self.inputList = inputs
        self.activationList = activations
        if self.inputList.size == 0:
            self.inputList = np.zeros(numUnits)
            
        if self.activationList.size == 0:
            self.activationList = np.zeros(numUnits)
        
    def checkWeights(self, numWeights): #is the length of the weight list passed in valid?
        if(self.weightList.size != 0 and self.weightList.size != numWeights):
            raise ValueError("The weight list you passed in has a length that doesn't match the number of weights you specified--the weightlist is size " + str(self.weightList.size) + " and the number of weights you specified is " + str(numWeights))
            
class RandomNetwork(Module):
    def __init__(self, nInput, nHidden, nOutput, predicted):
        numWeights = nInput * nHidden + nHidden * nOutput
        weightList = np.random.rand(numWeights)
        lam = 1
        beta = 0
        numUnits = nInput + nHidden + nOutput
        super().__init__(numUnits,weightList,np.zeros(numUnits),np.random.rand(numUnits),beta,lam)
                                                                #random activation values to start
        c = 0 #counter for location in the weightlist
        for i in range(nInput): #fill in input -> hidden connections
            for k in range(nHidden):
                hiddenIndex = k+nInput
                self.weightMatrix[i][hiddenIndex] = weightList[c] 
                c += 1
            
        for i in range(nHidden):
            for k in range(nOutput):
                hiddenIndex = i+nInput
                outputIndex = k+nInput+nHidden
                self.weightMatrix[hiddenIndex][outputIndex] = weightList[c]
                c += 1
                
def actFPrime(x):
    return (L*np.exp(-1*L*(x-B)))/(np.power((np.exp(1/(1+np.exp(x)))+1),2)) #derivative of the activation function
