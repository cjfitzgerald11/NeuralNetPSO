import math
class Node:
    def __init__(self, typeNode):
        self.nodeTypes = {"input": 0, "bias": 1, "output": 2}
        self.nodeType = self.nodeTypes[typeNode]

    """Activation method for a node. Input nodes return their value, bias nodes
    return 1, and output nodes return the sigmoid value of their input."""
    def activate(self, input):
        if self.nodeType == 0:
            return input
        elif self.nodeType == 1:
            return 1
        else:
            return self.sigmoid(input)

    """Returns the sigmoid value for a given sum."""
    def sigmoid(self,sum):
        try:
            return 1.0/(1.0 + math.exp(-sum + 0.5))
        except:
            return 0

    """Returns the value of the derivative of the sigmoid method; used in
    backprogation training."""
    def g_prime(self,sum):
        try:
        #g_dot = math.exp(0.5 - activation) / (((math.exp(0.5 - activation)) + 1) ** 2)
            g_dot = (1 - self.sigmoid(sum)) * self.sigmoid(sum)
            return g_dot
        except:
            return 0
