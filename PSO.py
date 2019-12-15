import sys
from Particle import Particle
from Neighborhood import Neighborhood
from Perceptron import Perceptron
import statistics
import time

class PSO:
    def __init__(self,dimension,trainTester,numIterations,seedRadius,seedVelocity):
        self.topology = 'ri'
        self.sizeSwarm = int(8)
        self.numIterations = int(numIterations)
        self.dimension = int(dimension)
        self.globalBestLocation = []
        self.globalBestValue = 10000000
        self.particles = []
        self.seedRadius = seedRadius
        self.seedVelocity = seedVelocity
        self.NH = Neighborhood(self.topology,int(dimension))
        self.trainTeser = trainTester

    #initialize a swarm of size self.sizeSwarm with randomly located particles
    def buildSwarm(self,weights):
        #build sizeSwarm particles, by defualt they are initialized randomly
        for i in range(self.sizeSwarm):
            p = Particle(self.dimension,self.trainTeser)
            p.seedInit(weights,self.seedRadius,self.seedVelocity)
            self.particles += [p]
        #check for global best
        self.updateGlobalBest()

    #this method will individiually update partciles based on their personal
    #best location and their neighborhood best location
    def updateSwarm(self):
        for particle in self.particles:
            curIndex = self.particles.index(particle)
            nhBest = self.NH.getBestNeighbor(self.particles,particle.getFunctionValue(),particle.getLocation(),curIndex)
            particle.updateLocation(nhBest)

    #check to see if terminantion condition is met (found 0.0 function value)
    def minFound(self):
        if self.globalBestValue == 0.0:
            return True
        else:
            return False

    #this funciton will format the output of the program,
    #returning the minimum value found and the location of the
    #minimum value
    def formatOutput(self, numIterations, values):
        return {"val" :  self.globalBestValue, "location" : self.globalBestLocation, "iterations" : numIterations, "intervalValues": values}

    #this method will look through all particles and find the global best
    def updateGlobalBest(self):
        for particle in self.particles:
            if particle.pBestValue() < self.globalBestValue:
                self.globalBestValue = particle.pBestValue()
                self.globalBestLocation = particle.pBest

    #this function will run numIterations of the PSO algorithm and print
    #the minimum value found and the minimum location after the iterations.
    #Also if a 0.0 value is found it will break out of the loop and return
    #the location
    def run(self):
        for i in range(self.numIterations):
            self.updateSwarm()
            self.updateGlobalBest()
            if self.minFound():
                break
        return self.globalBestLocation
