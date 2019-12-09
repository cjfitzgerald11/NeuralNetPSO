import random

class Particle:
    def __init__(self, dimension, function, funcType):
        self.dimension = dimension
        #used to initialize starting velocity and position
        self.funcType = funcType
        #the location of the particle
        self.location = []
        #particle velocity
        self.velocity = []
        #personal best found, initialize as current position
        self.pBest = self.location
        #evaluation function
        self.function = function
        #personal best acceleration coefficient
        self.phi1 = 2.05
        #global best acceleration coefficient
        self.phi2 = 2.05
        #constriction factor
        self.constrictionFactor = 0.7298

    def __str__(self):
        return self.location

    def randomInit(self):
        self.initPosition()
        self.initVelocity()

    def initPosition(self):
        rng = []
        rng = (-1.0, 1.0)
        # if self.funcType == "rok":
        #     rng = (15.0, 30.0)
        # elif self.funcType == "ras":
        #     rng = (16.0, 32.0)
        # else:
        #     rng = (2.56, 5.12)
        for i in range(self.dimension):
            self.location += [random.uniform(*rng)]

    def initVelocity(self):
        rng = []
        rng = (-1.0, 1.0)
        # if self.funcType == "rok":
        #     rng = (-2.0, 2.0)
        # elif self.funcType == "ras":
        #     rng = (-2.0, 4.0)
        # else:
        #     rng = (-2.0, 4.0)
        for i in range(self.dimension):
            self.velocity += [random.uniform(*rng)]

    #getter method for location
    def getLocation(self):
        return self.location

    #setter method for location
    def setLocation(self,location):
        self.location = location

    #getter method for personal best
    def pBest(self):
        return self.pBest

    def pBestValue(self):
        return self.function.eval(self.pBest)

    #getter method for function value at current position
    def getFunctionValue(self):
        return self.function.eval(self.location)

    def updateLocation(self,nhBest):
        pbAc = self.pBestAcceleration()
        nbAc = self.nBestAcceleration(nhBest)
        self.updateVelocity(pbAc,nbAc)
        for i in range(len(self.velocity)):
            self.location[i] += self.velocity[i]
        self.updatePersonalBest()

    #compute the acceleration due to personal best
    def pBestAcceleration(self):
        pbAc = []
        i = 0
        #compute acceleration as difference between current location and personal
        #best location in self.dimensions dimenions
        for pbi in self.pBest:
            iAc = pbi - self.location[i]
            iAc = iAc*self.phi1*random.random()
            pbAc += [iAc]
            i += 1
        return pbAc

    #compute the acceleration due to neighborhood best
    def nBestAcceleration(self,nhBest):
        nbAc = []
        i = 0
        #compute acceleration as difference between current location and neighborhood
        #best location in self.dimensions dimenions
        for nbi in nhBest:
            iAc = nbi - self.location[i]
            iAc = iAc*self.phi2*random.random()
            nbAc += [iAc]
            i += 1
        return nbAc

    #compute the velocity vector given personal best acceleration and
    #neighborhood best acceleration
    def updateVelocity(self,pbAc,nbAc):
        #constrict the new velocity and reset the current velocity
        for i in range(len(pbAc)):
            self.velocity[i] += pbAc[i] + nbAc[i]
            #constrict
            self.velocity[i] = self.velocity[i] * self.constrictionFactor

    def updatePersonalBest(self):
        currentFuncVal = self.getFunctionValue()
        pBestFuncVal = self.function.eval(self.pBest)
        if currentFuncVal < pBestFuncVal:
            self.pBest = self.location
