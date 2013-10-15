"""Network administrator task."""

from Tools import *
from Domain import Domain
import os
from Tools import __rlpy_location__

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Robert H. Klein", "Alborz Geramifard"]


class SystemAdministrator(Domain):
    """
    **System Administrator Problem**
    
    Computers in a network randomly fail and influence the probability
    of connected machines failing as well - the system administrator must work
    to keep as many machines running as possible, but she can only fix one
    at a time.
    
    | **State**
    | Each computer has binary state {BROKEN, RUNNING}.
    | The state space is thus *2^n*, where *n* is the number of computers in the system.
    
    | *Example*
    | [1 1 0 1] -> computers 0,1,3 are RUNNING, computer 2 is BROKEN.

    | **Actions**
    The action space is the integers [0,n], where 0 corresponds to taking
    no action, and [1,n] selects a computer to repair.
    
    Repairing a computer causes its state to become RUNNING regardless of its
    previous state.
    
    However, penalty -0.75 is applied for taking a repair action.
    
    
    | **Visualization**
    Broken computers are colored red, and any links to other computers
    change from solid to dotted, reflecting the higher probability 
    of failure of those machines.
    
    | **Reference**
    For details, see:
    
        Carlos Guestrin, Daphne Koller, Ronald Parr, and Shobha Venkataraman.
        Efficient Solution Algorithms for Factored MDPs.  Journal of Artificial
        Intelligence Research (2003) Issue 19, p 399-468.
    """
    NEIGHBORS       = [] # Each cell corresponds to a computer; contents of cell is a list of neighbors connected to that computer
    UNIQUE_EDGES    = [] # A list of tuples (node1, node2) where node1 and node2 share an edge and node1 < node2.

    P_SELF_REPAIR   = 0.04 #: Probability of a machine randomly self-repairing (no penalty)
    P_REBOOT_REPAIR = 1.0 #: Probability that a machine becomes RUNNING after a repair action is taken

    IS_RING         = False #: For ring structures, Parr enforces assymetry by having one machine get extra reward for being RUNNING.

    REBOOT_REWARD   = -0.75 #: Penalty applied for a REPAIR action
    # Computer "up" reward implicitly 1; tune other rewards relative to this.

    episodeCap      = 200        #: Maximum number of steps
    gamma           = .95        #: Discount factor

    # Plotting Variables
    networkGraph    = None     # Graph of network used for visualization
    networkPos  = None       # Position of network graph

    # Possible values for each computer
    BROKEN, RUNNING = 0,1
    _NUM_VALUES = 2         # Number of values possible for each state, must be hand-coded to match number defined above

    default_map_dir = os.path.join(__rlpy_location__, "Domains", "SystemAdministratorMaps")

    def __init__(self, networkmapname=os.path.join(default_map_dir, "20MachTutorial.txt"),
                 logger=None):
        """
        :param networkmapname: The name of the file to use as the computer
            network map.  Assumed to be located in the SystemAdministratorMaps
            directory of RLPy.
        """
        path                    = networkmapname
        self.IS_RING            = "ring.txt" in networkmapname.lower()
        self.loadNetwork(path)
        # TODO Need a check here for degenerate
        self.actions_num            = self.computers_num + 1     # Number of Actions, including no-op
        self.statespace_limits      = tile([0,self._NUM_VALUES-1],(self.computers_num,1))# Limits of each dimension of the state space. Each row corresponds to one dimension and has two elements [min, max]
        super(SystemAdministrator,self).__init__(logger)
        if self.logger:
            self.logger.log('Computers:\t%d' % self.computers_num)
            self.logger.log('Edges:\t\t%s' % str(self.UNIQUE_EDGES))
            self.logger.log('Neighbors:')
            for i in range(self.computers_num):
                self.logger.log('%d : %s' % (i,str(list(self.NEIGHBORS[i]))))

    
    def loadNetwork(self, path):
        """
        :param path: Path to the map file, of form
            \'/Domains/SystemAdministratorMaps/mapname.txt\'
        Sets the internal variables _Neighbors and _Edges, where each cell of
        _Neighbors is a list containing the neighbors of computer node <i>
        at index <i>, and
        _Edges is a list of tuples (node1, node2) where node1 and node2
        share an edge and node1 < node2.

        """
        _Neighbors = []
        f = open(path, 'rb')
        reader = csv.reader(f, delimiter=',')
        self.computers_num = 0
        for row in reader:
            row = map(int,row)
            _Neighbors.append(row)
            self.computers_num = max(max(row)+1,self.computers_num)
        self.setUniqueEdges(_Neighbors)
        self.setNeighbors()
    def showDomain(self, a=0):
        s = self.state
        if self.networkGraph is None: #or self.networkPos is None:
            self.networkGraph = nx.Graph()
            # enumerate all computer_ids, simulatenously iterating through neighbors list and compstatus
            for computer_id, (neighbors, compstatus) in enumerate(zip(self.NEIGHBORS,s)):
                self.networkGraph.add_node(computer_id, node_color = "w") # Add a node to network for each computer
            for uniqueEdge in self.UNIQUE_EDGES:
                    self.networkGraph.add_edge(uniqueEdge[0],uniqueEdge[1], edge_color = "k") # Add an edge between each neighbor
            self.networkPos = nx.circular_layout(self.networkGraph)
            nx.draw_networkx_nodes(self.networkGraph, self.networkPos, node_color="w")
            nx.draw_networkx_edges(self.networkGraph, self.networkPos, edges_color="k")
            nx.draw_networkx_labels(self.networkGraph, self.networkPos)
            pl.show()
        else:
            pl.clf()
            blackEdges = []
            redEdges = []
            greenNodes = []
            redNodes = []
            for computer_id, (neighbors, compstatus) in enumerate(zip(self.NEIGHBORS,s)):
                if(compstatus == self.RUNNING):
                    greenNodes.append(computer_id)
                else:
                    redNodes.append(computer_id)
            for uniqueEdge in self.UNIQUE_EDGES: # Iterate through all unique edges
                if(s[uniqueEdge[0]] == self.RUNNING and s[uniqueEdge[1]] == self.RUNNING):
                    # Then both computers are working
                    blackEdges.append(uniqueEdge)
                else: # If either computer is BROKEN, make the edge red
                    redEdges.append(uniqueEdge)
            # "if redNodes", etc. - only draw things in the network if these lists aren't empty / null
            if redNodes:    nx.draw_networkx_nodes(self.networkGraph, self.networkPos, nodelist=redNodes, node_color="r",linewidths=2)
            if greenNodes:  nx.draw_networkx_nodes(self.networkGraph, self.networkPos, nodelist=greenNodes, node_color="w",linewidths=2)
            if blackEdges:  nx.draw_networkx_edges(self.networkGraph, self.networkPos, edgelist=blackEdges, edge_color="k",width=2,style='solid')
            if redEdges:    nx.draw_networkx_edges(self.networkGraph, self.networkPos, edgelist=redEdges, edge_color="k",width=2,style='dotted')
        nx.draw_networkx_labels(self.networkGraph, self.networkPos)
        pl.draw()

    def step(self,a):
        #ns = s[:] # make copy of state so as not to affect original mid-step
        ns = self.state.copy()
       # print 'action selected',a,s
        totalRebootReward = 0
        for computer_id, compstatus in enumerate(self.state):
            if(a == computer_id): #Reboot action on this computer
                totalRebootReward += self.REBOOT_REWARD
                # NOTE can break up if-statement below to separate cases
                if (self.random_state.random_sample() <= self.P_REBOOT_REPAIR):
                    ns[computer_id] = self.RUNNING
                else:
                    ns[computer_id] = self.BROKEN
            else: # Transition to new state probabilistically
                if (compstatus == self.RUNNING):
                    # take neighbors of computer_id and sum over each of their current values
                    sumOfNeighbors = sum([self.state[i] for i in self.NEIGHBORS[computer_id]])
                    # TODO this expression should be a function, or something
                    p_broken = 1.0 - (0.45 + 0.5 * (1+sumOfNeighbors) / (1+len(self.NEIGHBORS[computer_id])))
                    if(self.random_state.random_sample() < p_broken ):
                        ns[computer_id] = self.BROKEN
                else:
                    if(self.random_state.random_sample() < self.P_SELF_REPAIR):
                        ns[computer_id] = self.RUNNING
        if (self.IS_RING and s[0] == self.RUNNING): totalRebootReward += 1 # Per Guestrin, Koller, Parr 2003, rings have enforced asymmetry on one machine
        terminal = False
        self.state = ns.copy()
        return sum(self.state)+totalRebootReward,ns,terminal, self.possibleActions()
        # Returns the triplet [r,ns,t] => Reward, next state, isTerminal

    def s0(self):
        self.state = array([self.RUNNING for dummy in arange(0,self.state_space_dims)]) # Omits final index
        return self.state.copy(), self.isTerminal(), self.possibleActions()

    def possibleActions(self):
        s = self.state
        possibleActs = [computer_id for computer_id,compstatus in enumerate(s) if compstatus == self.BROKEN]
        possibleActs.append(self.computers_num) # append the no-op action
        return array(possibleActs)

    def setUniqueEdges(self, neighborsList):
        """
        :param neighborsList: each element at index i is a list of nodes
            connected to the node at i.
        Constructs a list (node1, node2) where node1 and node2 share an edge
        and node1 < node2 and sets the unique Edges of the network
        (all edges are bidirectional).
        """
        self.UNIQUE_EDGES = []
        for computer_id, neighbors in enumerate(neighborsList):
            for neighbor_id in neighbors:
                edge = (min(neighbor_id,computer_id), max(neighbor_id,computer_id))
                found = [t for t in self.UNIQUE_EDGES if t[0] == edge[0] and t[1] == edge[1]]
                if found == []:
                    self.UNIQUE_EDGES.append(edge)
    def setNeighbors(self):
        """
        Sets the internal NEIGHBORS variable
        
        .. note::

            Requires a call to setUniqueEdges() first.

        """
        self.NEIGHBORS = {} # Initialize list so we don't get out of bounds errors
        for edgePair in self.UNIQUE_EDGES:
            # Add each node as a neighbor to each other
            s,d = edgePair
            if s in self.NEIGHBORS:
                self.NEIGHBORS[s] += [d]
            else:
                self.NEIGHBORS[s] = [d]
            if d in self.NEIGHBORS:
                self.NEIGHBORS[d] += [s]
            else:
                self.NEIGHBORS[d] = [s]
        for i in range(self.computers_num):
            self.NEIGHBORS[i] = array(self.NEIGHBORS[i])
