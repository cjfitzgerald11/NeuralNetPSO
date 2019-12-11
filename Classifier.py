from Perceptron import Perceptron
from Test import Test
from ImageReader import ImageReader
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PSO import PSO

class Classifier:
    def __init__(self,learningRate, epochs, imageSize,numBiasNodes,numOuputNodes):
        self.learningRate = float(learningRate)
        self.epochs = int(epochs)
        self.imageSize = int(imageSize)
        self.numBiasNodes = int(numBiasNodes)
        self.numOuputNodes = int(numOuputNodes)
        self.trainTester = []
        self.Tester = []
        self.testResults = []
        self.runTimes = []
        self.psoOptimizer = []

    def run(self):
        TrainImages,TrainAnswers,TestImages,TestAnswers = self.getImageSets()
        #build defulat perceptron
        Percep = Perceptron(self.imageSize**2, self.numBiasNodes,self.numOuputNodes,self.learningRate)
        Percep.init()
        self.Tester = Test(TestImages,TestAnswers,Percep)
        self.trainTester = Test(TrainImages,TrainAnswers,Percep)
        dimension = (self.imageSize*self.imageSize + self.numBiasNodes)*self.numOuputNodes
        self.psoOptimizer = PSO(dimension,self.Tester,self.trainTester)
        formatedOutput, testResults, timeResults = self.psoOptimizer.run()
        return testResults, timeResults

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
numBiasNodes = sys.argv[4]
numOuputNodes = sys.argv[5]

numRuns = 5
classify = Classifier(learningRate,epochs,imageSize,numBiasNodes,numOuputNodes)
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
