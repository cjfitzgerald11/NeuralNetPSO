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

    def trainWeights(self,trainImage,trainAnswer):
        evaluation = self.evaluate(trainImage)
        activations,sums = self.getActivations(trainImage)
        outputNode = 0
        for activation in activations:
            if len(activations) == 1:
                activation = activation*10
                self.adjustWeight(outputNode, activation, sums[outputNode], trainImage, trainAnswer)
            else:
                if outputNode == trainAnswer:
                    self.adjustWeight(outputNode, activation,  sums[outputNode],trainImage,  1)
                else:
                    self.adjustWeight(outputNode, activation,  sums[outputNode], trainImage, 0)
            outputNode += 1

    def adjustWeight(self,outputNode, solution, sum, trainImage, answer):
        err = (answer - solution)/10
        g_prime = self.inputNode.g_prime(sum)
        for inputNode in range(len(self.inputNodes)):
            currentWeight = self.getGraphWeight(inputNode,outputNode)
            updatedWeight = currentWeight  + (err*g_prime*self.learningRate*(trainImage[inputNode] + .1))
            self.setGraphWeight(inputNode, outputNode,updatedWeight)
