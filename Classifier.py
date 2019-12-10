from Perceptron import Perceptron
from Train import Train
from Test import Test
from ImageReader import ImageReader
import sys

class Classifier:
    def __init__(self,learningRate, epochs, imageSize,numBiasNodes,numOuputNodes):
        self.learningRate = float(learningRate)
        self.epochs = int(epochs)
        self.imageSize = int(imageSize)
        self.numBiasNodes = int(numBiasNodes)
        self.numOuputNodes = int(numOuputNodes)
        self.Trainer = []
        self.Tester = []

    def run(self):
        TrainImages,TrainAnswers,TestImages,TestAnswers = self.getImageSets()
        self.Trainer = Train(TrainImages,TrainAnswers,self.learningRate)
        self.Tester = Test(TestImages,TestAnswers)
        #build defulat perceptron
        Percep = Perceptron(self.imageSize**2, self.numBiasNodes,self.numOuputNodes,self.learningRate)
        Percep.init()
        i = 0
        while i < self.epochs:
            print("--------")
            print("epoch: ", i)
            print("--------")
            Percep,TestResults = self.epoch(Percep)
            print(TestResults)
            i += 1

    def epoch(self,perceptron):
        TrainedPerceptron = self.Trainer.train(perceptron)
        TestResults = self.Tester.test(TrainedPerceptron)
        return TrainedPerceptron,TestResults

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

#Get Terminal Input
learningRate = sys.argv[1]
epochs = sys.argv[2]
imageSize = sys.argv[3]
numBiasNodes = sys.argv[4]
numOuputNodes = sys.argv[5]

classify = Classifier(learningRate,epochs,imageSize,numBiasNodes,numOuputNodes)
classify.run()
