from Perceptron import Perceptron
from Node import Node

class Train:
    def __init__(self, learningRate):
        self.learningRate = learningRate

    def train(self,perceptron, trainImages,trainAnswers):
        for i in range(len(trainImages)):
            perceptron.trainWeights(trainImages[i],trainAnswers[i])
        return perceptron
