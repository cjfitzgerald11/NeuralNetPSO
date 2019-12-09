from Perceptron import Perceptron

class Test:
    def test(self,perceptron, testImages,testAnswers):
        print("--------------")
        print("TEST")
        print("--------------")
        numSuccess = 0
        numTests = len(testImages)
        for i in range(len(testImages)):
            testImage = testImages[i]
            testAnswer = testAnswers[i]
            prediction = perceptron.evaluate(testImage)
            print("prediction: ", prediction)
            print("answer: ", testAnswer)
            if prediction == testAnswer:
                numSuccess += 1
        return numSuccess/numTests
