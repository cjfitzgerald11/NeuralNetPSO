import sys
from Particle import Particle
from Function import Function
from Neighborhood import Neighborhood
import statistics

class PSO:
    def __init__(self, topology, sizeSwarm, numIterations, funcType, dimension):
        self.topology = topology
        self.sizeSwarm = int(sizeSwarm)
        self.numIterations = int(numIterations)
        self.funcType = funcType
        self.function = Function(funcType,int(dimension))
        self.dimension = int(dimension)
        self.globalBestLocation = []
        self.globalBestValue = 10000000
        self.particles = []
        self.NH = Neighborhood(topology,int(dimension))

    #initialize a swarm of size self.sizeSwarm with randomly located particles
    def buildSwarm(self):
        #build sizeSwarm particles, by defualt they are initialized randomly
        for i in range(self.sizeSwarm):
            p = Particle(self.dimension,self.function,self.funcType)
            p.randomInit()
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
        self.buildSwarm()
        vals = []
        for i in range(self.numIterations):
            self.updateSwarm()
            self.updateGlobalBest()
            if self.minFound():
                break
            if i % 1000 == 0:
                vals += [self.globalBestValue]
        return self.formatOutput(i,vals)

#Get paramater input from command line (topology, sizeSwarm, numIterations, function, dimension)
topology = sys.argv[1]
sizeSwarm = sys.argv[2]
numIterations = sys.argv[3]
function = sys.argv[4]
dimension = sys.argv[5]

eA = PSO(topology,sizeSwarm,numIterations,function,dimension)
res = eA.run()
print("Minimum Value Found: ", res["val"])
print("Minimum Value Location: ", res["location"])
print("Number of Iterations: ", int(res["iterations"]) + 1)


#Used to run the program for multiple iterations
"""
#computes the median function value of each function for all 20 runs at each interval
def computeMedianFunctionVal(vals):
    valsOfI = []
    for row in range(len(vals)):
        for interval in range(len(vals[row])):
            if interval > (len(valsOfI) -1):
                valsOfI += [[vals[row][interval]]]
            else:
                valsOfI[interval] += [vals[row][interval]]
    medianValsOfI = [statistics.median(i) for i in valsOfI]
    return medianValsOfI


nunRuns = 20
mins = []
locations = []
iterations = []
intervalValues = []
for i in range(nunRuns):
    eA = PSO(topology,sizeSwarm,numIterations,function,dimension)
    res = eA.run()
    min = res["val"]
    location = res["location"]
    iteration = res["iterations"]
    iValues = res["intervalValues"]
    mins += [min]
    locations += [location]
    iterations += [iteration]
    intervalValues += [iValues]
print("topology: ", topology)
print("sizeSwarm: ", sizeSwarm)
print("function: ", function)
print("dimension: ", dimension)
print("median min: ", statistics.median(mins))
print("mean min: ", statistics.mean(mins))
print("average iterations: ", statistics.mean(iterations))
print("median value for each 1000 iteration: ", computeMedianFunctionVal(intervalValues))
"""
