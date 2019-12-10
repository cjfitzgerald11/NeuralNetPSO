from Perceptron import Perceptron

class Test:
    def __init__(self,testImages,testAnswers):
        self.testImages = testImages
        self.testAnswers = testAnswers

    def test(self,perceptron):
        print("--------------")
        print("TEST")
        print("--------------")
        numSuccess = 0
        numTests = len(self.testImages)
        for i in range(numTests):
            testImage = self.testImages[i]
            testAnswer = self.testAnswers[i]
            prediction = perceptron.evaluate(testImage)
            if prediction == testAnswer:
                numSuccess += 1
        return numSuccess/numTests
