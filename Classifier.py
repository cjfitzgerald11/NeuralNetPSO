from Perceptron import Perceptron
from Test import Test
from ImageReader import ImageReader
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PSO import PSO
from Train import Train

class Classifier:
    def __init__(self,learningRate, epochs, imageSize,numOuputNodes,numPSOIterations,psoSeedRadius,psoSeedVelocity):
        self.learningRate = float(learningRate)
        self.epochs = int(epochs)
        self.imageSize = int(imageSize)
        self.numOuputNodes = int(numOuputNodes)
        self.numBiasNodes = 0
        self.testResults = []
        self.runTimes = []
        self.numPSOIterations = int(numPSOIterations)
        self.PSOSeedRadius = float(psoSeedRadius)
        self.psoSeedVelocity = float(psoSeedVelocity)


    def run(self):
        TrainImages,TrainAnswers,TestImages,TestAnswers = self.getImageSets()
        #build defulat perceptron
        Percep = Perceptron(self.imageSize**2, self.numBiasNodes,self.numOuputNodes,self.learningRate)
        Percep.init()
        Tester = Test(TestImages,TestAnswers,Percep)
        trainTester = Test(TrainImages,TrainAnswers,Percep)
        dimension = (self.imageSize*self.imageSize + self.numBiasNodes)*self.numOuputNodes
        trainer = Train( TrainImages,TrainAnswers, self.learningRate,self.numPSOIterations,self.PSOSeedRadius,self.psoSeedVelocity)
        self.testResults = []
        self.runTimes = []
        startTime = time.process_time()
        i = 0
        while i < self.epochs:
            Percep,TestResults = self.epoch(Percep,dimension,trainTester,trainer,Tester)
            self.testResults += [TestResults]
            currentTime = time.process_time() - startTime
            self.runTimes += [currentTime]
            i += 1
        return  self.testResults,self.runTimes

    def epoch(self,perceptron,dimension,trainTester,trainer,tester):
        trainedPerceptron = trainer.trainBackProp(perceptron)
        trainedWeights = trainedPerceptron.perceptronGraph
        psoTrainedWeights = trainer.trainPSO(dimension,trainTester,trainedWeights,self.PSOSeedRadius)
        trainedPerceptron.updateAllWeights(psoTrainedWeights)
        TestResults = tester.test(trainedPerceptron)
        return trainedPerceptron,TestResults

    def getImageSets(self):
        trainFile = "files/train" + str(imageSize) + ".txt"
        testFile = "files/test" + str(imageSize) + ".txt"
        TrainReader = ImageReader(trainFile,self.imageSize)
        TestReader = ImageReader(testFile,self.imageSize)
        #only reads images if it hasn't been called before
        TrainReader.readImages()
        TestReader.readImages()
        TrainImages,TrainAnswers = TrainReader.getImages()
        TestImages,TestAnswers = TestReader.getImages()
        return TrainImages,TrainAnswers,TestImages,TestAnswers

    def graphResults(self):
        #make test results percents
        testResults = [100*result for result in self.testResults]
        plt.plot(self.runTimes,testResults)
        plt.title("Run Time (s) vs. % Correct")
        plt.xlabel("Run Time (s)")
        plt.ylabel("% Correct")
        plt.show()
        plt.cla()

        epochs = np.arange(1,self.epochs + 1)
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(epochs,testResults)
        plt.title("Epochs vs. % Correct")
        plt.xlabel("Epoch")
        plt.ylabel("% Correct")
        plt.show()

#Get Terminal Input
learningRate = sys.argv[1]
epochs = sys.argv[2]
imageSize = sys.argv[3]
numOuputNodes = sys.argv[4]
numPSOIterations = sys.argv[5]
psoSeedRadius = sys.argv[6]
psoSeedVelocity = sys.argv[7]

numRuns = 2
classify = Classifier(learningRate,epochs,imageSize,numOuputNodes,numPSOIterations,psoSeedRadius,psoSeedVelocity)
allTestResults = []
allTimeResults = []
for i in range(numRuns):
    print("run: ", i + 1)
    testResults, timeResults = classify.run()
    allTestResults += [testResults]
    allTimeResults += [timeResults]

averageTest = np.mean(allTestResults, axis=0)
averageTest = [100*result for result in averageTest]
print("Average Test Success: ", averageTest)
print("maxValue: ", max(averageTest))
averateTime = np.mean(allTimeResults, axis=0)
print("Average Time: ", averateTime)
