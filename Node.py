import math
class Node:
    def __init__(self, typeNode):
        self.nodeTypes = {"input": 0, "bias": 1, "output": 2}
        self.nodeType = self.nodeTypes[typeNode]

    def activate(self, input):
        if self.nodeType == 0:
            return input
        elif self.nodeType == 1:
            return 1
        else:
            return 1.0/(1.0 + pow(math.e,(-input + 0.5)))

    def g_prime(self,activation):
        g_dot = (math.e ** (0.5 - activation)) / (((math.e ** (0.5 - activation)) + 1) ** 2)
        return g_dot
