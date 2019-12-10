import random
import numpy as np

class Neighborhood:
    def __init__(self, neighborhoodType, dimension):
        self.neighborhoodType = neighborhoodType
        self.dimension = dimension
        self.neighbors = []


    """This helper method initializes a neighborhood for a given problem type."""
    def initNeighbors(self, particles, curIndex):
        if self.neighborhoodType == 'gl':
            self.neighbors = particles
        elif self.neighborhoodType == 'ri':
            self.riNeighbors(particles, curIndex)
        elif self.neighborhoodType == 'vn':
            self.vnNeighbors(particles, curIndex)
        elif self.neighborhoodType == 'ra':
            self.raNeighbors(particles, curIndex)


    """Given all the particles in the swarm and the location
    of a particle, it will return the location of the best
    particle in its neighborhood."""
    def getBestNeighbor(self, particles, curScore, position, curIndex):
        if len(self.neighbors) == 0:
            self.initNeighbors(particles, curIndex)
        if self.neighborhoodType== 'gl':
            return self.getBestLocation(self.neighbors, curScore, position)
        elif self.neighborhoodType== 'ri':
            return self.getBestLocation(self.neighbors, curScore, position)
        elif self.neighborhoodType== 'vn':
            return self.getBestLocation(self.neighbors, curScore, position)
        elif self.neighborhoodType== 'ra':
            self.raNeighbors(particles, curIndex)
            return self.getBestLocation(self.neighbors, curScore, position)
        else:
            return position

    """Helper method that returns the best neighbor when given a neighborhood,
    the evaluation of the current particle's location, and the current particle's
    position."""
    def getBestLocation(self, neighborhood, curScore, position):
        loBest = curScore
        loBestLocation = position
        #print("neighbors: ", self.neighbors)
        for neighbor in self.neighbors:
            neighborVal = neighbor.getFunctionValue()
            if neighborVal < loBest:
                loBest = neighborVal
                loBestLocation = neighbor.getLocation()
        return loBestLocation



    """Helper method used to create the ring swarm neighborhood topology.
    Uses the list of particles, the evaluation of the current particle, the
    current particles position, and the current particle's index to
    create the neighborhood."""
    def riNeighbors(self, particles, curIndex):
        if curIndex == (len(particles) - 1):
            self.neighbors.append(particles[curIndex - 1])
            self.neighbors.append(particles[0])
        elif curIndex <= (len(particles) - 2):
            self.neighbors.append(particles[curIndex - 1])
            self.neighbors.append(particles[curIndex + 1])



    """Helper methods used to find the neighborhood in von Neumann swarm topology."""
    def vnNeighborsHelper(self, numPart, particles):
        width = 0
        height = 0
        if numPart == 16:
            width = 4
            height = 4
        elif numPart == 30:
            width = 5
            height = 6
        else:
            width = 7
            height = 7
        topology = [[None] * width] * height
        i = 0
        while i < numPart - 1:
            #modulo population of the topology
            xPos = 0
            while xPos < width:
                yPos = i % width
                topology[xPos][yPos] = particles[i]
                xPos++
                i++
        return (topology, width, height)

    """Helper method used to determine a particle's neighbors in Von Neumann
    topology. Uses the list of particles and the current particle's index to
    update the self.neighbors instance variable. Calls vnNeighborsHelper to do so."""
    def vnNeighbors(self, particles, curIndex):
        neighborhood = self.vnNeighborsHelper(len(particles), particles)
        topology = neighborhood[0]
        width = curIndex % neighborhood[1]
        height = curIndex // neighborhood[2]
        #Add left & above neighbors to neighbors list
        self.neighbors.append(topology[width - 1][height])
        self.neighbors.append(topology[width][height - 1])
        #get neighbor to the right
        if width + 1 <= (neighborhood[1] - 2):
            self.neighbors.append(topology[width + 1][height])
        else:
            self.neighbors.append(topology[0][height])
        #get neighbor below
        if height + 1 <= neighborhood[2] - 2:
            self.neighbors.append(topology[width][height + 1])
        else:
            self.neighbors.append(topology[width][0])


    """Helper method used to find the best neighbor in random swarm topology
    Uses the list of particles, and the current index to create a particles
    random neighborhood if the threshold criteria are met."""
    def raNeighbors(self, particles, curIndex):
        if random.random() <= 0.8 and len(self.neighbors) != 0:
            pass
        else:
            self.neighbors = []
            neighborhoodSize = 5
            neighborhood = set()
            neighbors = []
            indexes = []
            #Create a list of all indexes other than the curIndex
            for i in range(curIndex):
                indexes.append(i)
            for i in range(curIndex + 1, len(particles)):
                indexes.append(i)
            #Choose neighbor indexes
            while len(neighborhood) < neighborhoodSize:
                neighborhood.add(random.choice(indexes))
            neighborhood = list(neighborhood)
            for i in range(len(neighborhood) - 1):
                self.neighbors.append(particles[neighborhood[i]])
