from Perceptron import Perceptron
from Node import Node
from PSO import PSO

class Train:
    def __init__(self, trainImages, trainAsnwers, learningRate,numPSOIterations,PSOSeedRadius,psoSeedVelocity):
        self.learningRate = learningRate
        self.numPSOIterations = numPSOIterations
        self.PSOSeedRadius = PSOSeedRadius
        self.psoSeedVelocity = psoSeedVelocity
        self.trainImages = trainImages
        self.trainAsnwers = trainAsnwers


    """Backpropagation training method. Trains the perceptron against each image
    in the file."""
    def trainBackProp(self,perceptron):
        for i in range(len(self.trainImages)):
            perceptron.trainWeights(self.trainImages[i],self.trainAsnwers[i])
        return perceptron

    """PSO synaptic weight trainig method. Initializes a PSO swarm and selects the
    best set of weights found by the swarm."""
    def trainPSO(self,dimension,trainTester,seedWeights,radius):
        psoOptimizer = PSO(dimension,trainTester,self.numPSOIterations,self.PSOSeedRadius,self.psoSeedVelocity)
        psoOptimizer.buildSwarm(seedWeights)
        trainedWeights = psoOptimizer.run()
        return trainedWeights
