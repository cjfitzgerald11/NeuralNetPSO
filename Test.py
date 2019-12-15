from Perceptron import Perceptron

class Test:
    def __init__(self,testImages,testAnswers,perceptron):
        self.testImages = testImages
        self.testAnswers = testAnswers
        self.perceptron = perceptron

    """Test method that returns the percentage of correct evaluations that a
    perceptron made during that epoch."""
    def test(self,perceptron):
        numSuccess = 0
        numTests = len(self.testImages)
        for i in range(numTests):
            testImage = self.testImages[i]
            testAnswer = self.testAnswers[i]
            prediction = perceptron.evaluate(testImage)
            if prediction == testAnswer:
                numSuccess += 1
        return numSuccess/numTests


    """Method used in PSO training that updates all weights and uses the above
    method to test this set of weights."""
    def testWeights(self,weights):
        if len(weights) > 0:
            self.perceptron.updateAllWeights(weights)
            return self.test(self.perceptron)
        else:
            return 0
