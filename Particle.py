import random
from Perceptron import Perceptron
from Test import Test
import numpy as np

class Particle:
    def __init__(self, dimension,trainTester):
        self.dimension = dimension
        #the location of the particle
        self.location = []
        #particle velocity
        self.velocity = []
        #personal best found, initialize as current position
        self.pBest = []
        #personal best acceleration coefficient
        self.phi1 = 2.05
        #global best acceleration coefficient
        self.phi2 = 2.05
        #constriction factor
        self.constrictionFactor = 0.7298
        self.trainTester = trainTester


    def __str__(self):
        return self.location

    """Initializes a particle within the seed radius and velocity according to
    the starting set of weights passed in as a parameter."""
    def seedInit(self,weights,radius,velocity):
        self.initPosition(weights,radius)
        self.initVelocity(velocity)

    """"Initializes a particle's position within the seed radius range."""
    def initPosition(self,weights,radius):
        rng = []
        rng = (-radius, radius)
        weights = np.array(weights).flatten()
        for i in range(self.dimension):
            if i == 0 :
                self.location += [weights[i]]
            else:
                self.location += [weights[i] + random.uniform(*rng)]

    """Initializes a velocity vector for the particle within the seed velocity
    range."""
    def initVelocity(self,velocity):
        rng = []
        rng = (-velocity, velocity)
        for i in range(self.dimension):
            if i == 0 :
                self.velocity += [.01]
            else:
                self.velocity += [random.uniform(*rng)]

    """Getter method for location."""
    def getLocation(self):
        return self.location

    """Setter method for location."""
    def setLocation(self,location):
        self.location = location

    """Getter method for personal best."""
    def pBest(self):
        return self.pBest

    """Returns the personal best value associated with a set of weights in the
    perceptron graph."""
    def pBestValue(self):
        percentCorrect = self.trainTester.testWeights(self.pBest)
        return 1 - percentCorrect

    """Getter method for function value at current position."""
    def getFunctionValue(self):
        percentCorrect = self.trainTester.testWeights(self.location)
        return 1 - percentCorrect


    """Method that updates the location of a particlce based on the NH best
    acceleration.""""
    def updateLocation(self,nhBest):
        pbAc = self.pBestAcceleration()
        nbAc = self.nBestAcceleration(nhBest)
        self.updateVelocity(pbAc,nbAc)
        for i in range(len(self.velocity)):
            self.location[i] += self.velocity[i]
        self.updatePersonalBest()

    """Compute the acceleration due to personal best."""
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

    """Compute the acceleration due to neighborhood best."""
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

    """Compute the velocity vector given personal best acceleration and
    neighborhood best acceleration."""
    def updateVelocity(self,persBestAcc,neighBestAcc):
        #constrict the new velocity and reset the current velocity
        for i in range(len(persBestAcc)):
            self.velocity[i] += persBestAcc[i] + neighBestAcc[i]
            #constrict
            self.velocity[i] = self.velocity[i] * self.constrictionFactor

    """Method to update the swarm's personal best set of weights."""
    def updatePersonalBest(self):
        currentFuncVal = self.getFunctionValue()
        pBestFuncVal = self.pBestValue()
        if currentFuncVal < pBestFuncVal:
            self.pBest = np.array(self.location)
