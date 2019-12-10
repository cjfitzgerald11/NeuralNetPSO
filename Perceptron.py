from Node import Node
import numpy as np
import random
import math

class Perceptron:
    def __init__(self, numInputNodes,numBiasNodes, numOutputNodes, learningRate):
        self.inputNodes = [True]*numInputNodes
        self.inputNodes += [False]*numBiasNodes
        self.numOutputNodes = numOutputNodes
        self.perceptronGraph = np.zeros((numInputNodes + numBiasNodes,numOutputNodes))
        self.learningRate = learningRate
        self.inputNode = Node("input")
        self.outputNode = Node("output")
        self.biasNode = Node("bias")

    def init(self):
        #initialize perceptron to default values (ranodm number between -1 and 1)
        for outputNode in range(self.numOutputNodes):
            for inputNode in range(len(self.inputNodes)):
                weight = random.uniform(-0.15, .15)
                self.setGraphWeight(inputNode,outputNode,weight)

    def getGraphWeight(self,inputNode,outputNode):
        return self.perceptronGraph[inputNode,outputNode]

    def setGraphWeight(self,inputNode,outputNode,weight):
        self.perceptronGraph[inputNode,outputNode] = weight

    def evaluate(self,image):
        activations,sums = self.getActivations(image)
        if len(activations) == 1:
            activation = activations[0]
            digitEstimate = activation * 10
            digit = round(digitEstimate,0)
            digit = int(digit)
            return digit
        else:
            maxActivation = 0
            maxIndex = 0
            i = 0
            for activation in activations:
                if activation > maxActivation:
                    maxActivation = activation
                    maxIndex = i
                i += 1
            return maxIndex

    def getActivations(self,image):
        activations = []
        sums = []
        for outputNode in range(self.numOutputNodes):
            sum = 0
            for inputNode in range(len(self.inputNodes)):
                weight = self.getGraphWeight(inputNode,outputNode)
                isInputNode = self.inputNodes[inputNode]
                if isInputNode:
                    imageValue = image[inputNode]
                    activation = self.inputNode.activate(imageValue)
                    sum += weight*activation
                else:
                    activation  = self.biasNode.activate(0)
                    sum += weight*activation
            sums += [sum]
            activations += [self.outputNode.activate(sum)]
        return activations, sums

    """Given a list of weights (a location in the PSO search), this method
    updates all of the weights in the Perceptron graph."""
    def updateAllWeights(self, weights):
        weightsArray = np.reshape(weights,(len(self.inputNodes),self.numOutputNodes))
        self.perceptronGraph = weightsArray
