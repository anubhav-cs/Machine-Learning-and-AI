################################################################################
#                                                                              #
# Artificial Intelligence                                                      #
#                                                                              #
# Problem:  Berkeley Pacman AI Contest: Pacman Capture the Flag                #
#                                                                              #
# Author:   Anubhav Singh                                                      #
#                                                                              #
# References:                                                                  #
#   1. http://ai.berkeley.edu/contest.html                                     #
#                                                                              #
################################################################################

##### Required updates to the program #####
# 1. Create graph in python
# 2. Code Shortest Path Algorithm
# 3. Code betweenness calculation algorithm
# 4. Code approximation of enemy agents using particle filter
# 5. Develop bellman ford algorithm
# 6. Value Iteration

import itertools
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
from util import manhattanDistance
import numpy as np
from decimal import Decimal
import operator
import copy
import distanceCalculator
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Kunti', second = 'Kunti'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # Records shared betweent the two ally agents
  S = [[],[],[],[]]
  enemy_trap_coords = {}
  trap_coords = {}
  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex,S,enemy_trap_coords,trap_coords), \
                  eval(second)(secondIndex,S,enemy_trap_coords,trap_coords)]

##########
# Agents #
##########

######################
# Parent Agent Class #
######################
class Kunti(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
##################################################################
##################################################################
##################################################################
  prev_10_actions = []
  counter = 0
  randomise = False
# Implemention of value-iteration

  def get_VI_route(self, gameState, reward_matrix, val, my_pos, walls, c_time):
    S = reward_matrix.keys()
    # theta = 100
    V_i = {}
    V_f = {}
    n_s = {}

    ## #print walls

    # condition_not_met = True
    for i in range(70):
      for state in S:
        val = -1000000
        state_reward_matrix = dict(reward_matrix)
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
          next_state = (state[0]+dx, state[1]+dy)
          if not walls[next_state[0]][next_state[1]]:
            temp = state_reward_matrix[state]+ 0.9*V_i.setdefault(next_state,0)
            if temp > val:
              val = temp
        V_f[state] = (int)(val)
      V_i = dict(V_f)
      if time.time() - c_time > 0.8:
        break
    ## #print V_f

    for state in [my_pos]:
      val = -1000000
      state_reward_matrix = dict(reward_matrix)
      for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        next_state = (state[0]+dx, state[1]+dy)
        if not walls[next_state[0]][next_state[1]]:
          temp = state_reward_matrix[state]+ 0.9*V_f[next_state]
          ## #print temp, next_state
          if temp > val:
            val = temp
            n_s[state] = next_state
          elif temp == val:
            n_s[state] = random.choice([n_s[state], next_state])
    ## #print my_pos,"ACtion",n_s[my_pos]
    return n_s.setdefault(my_pos, my_pos)

########################################################
# init function re-defined to include the record S to be shared between agents
# this is used in particle filter to find approximate positions of enemy agents
# using the approximate manhattan distance of enemy agent

  def __init__( self, index, S,enemy_trap_coords, trap_coords, timeForComputing = .1 ):
    """
    Lists several variables you can query:
    self.index = index for this agent
    self.red = true if you're on the red team, false otherwise
    self.agentsOnTeam = a list of agent objects that make up your team
    self.distancer = distance calculator (contest code provides this)
    self.observationHistory = list of GameState objects that correspond
        to the sequential order of states that have occurred so far this game
    self.timeForComputing = an amount of time to give each turn for computing maze distances
        (part of the provided distance calculator)
    """
    self.S = S
    self.enemy_trap_coords = enemy_trap_coords
    self.trap_coords = trap_coords

    # Agent index for querying state
    self.index = index

    # Whether or not you're on the red team
    self.red = None

    # Agent objects controlling you and your teammates
    self.agentsOnTeam = None

    # Maze distance calculator
    self.distancer = None

    # A history of observations
    self.observationHistory = []

    # Time to spend each turn on computing maze distances
    self.timeForComputing = timeForComputing

    # Access to the graphics
    self.display = None


#######################################
# Particle Filter Approximate locator #
#######################################

###############################################

  def getApproxEnemyAgentPos(self, gameState):

    # get index of opponent's agents in an array
    opponent_indices = self.getOpponents(gameState)
    # initial number of particles
    num_particles = 20
    S=self.S

    my_index = self.index
    # get approx. manhattan distance of all agents
    noisy_dist = gameState.getAgentDistances()
    my_pos = gameState.getAgentPosition(self.index)

    for opp_index in opponent_indices:
      #time.sleep(5)
      opp_pos = gameState.getAgentPosition(opp_index)
      ## #print "########", self.index,opp_index, (opp_index==my_index-1 or opp_index==my_index+3)

      if opp_pos!=None:
        ## #print "@###@@##@@##@@##!@#@##$@#$@#$$#@@s"
        S[opp_index] = []
        for i in range(num_particles):
          S[opp_index].append((opp_pos, 1.0/num_particles))
      else :
        S[opp_index] = self.estimateUsingParticleFilter\
        (gameState, S[opp_index], noisy_dist[opp_index],\
        (opp_index==my_index-1 or opp_index==my_index+3), num_particles, opp_index)
    return S

# Util function to apply particle filter
  def estimateUsingParticleFilter(self, gameState, S_orig, noisy_dist, wasMoved, n, opp_index):
    num_particles = 20
    S_final = []

    # get the coordinates of observing agent
    my_pos = gameState.getAgentPosition(self.index)

    # normalization factor
    eps = 0

    # Essential particles
    E = {}
    E_i = []

    # Walls
    walls = gameState.getWalls()

    ## add coords and weights to E
    for k in range(len(S_orig)):
      E[S_orig[k][0]] = E.setdefault(S_orig[k][0],0)+ S_orig[k][1]

    # add next possible particles to essential particles
    for k in E.keys():
      E_i.append((k, E[k]))
      if wasMoved == True:
        for dx, dy in [(0,1), (0,-1), (-1,0), (1,0)]:
          if not walls[k[0]+dx][k[1]+dy]:
            E_i.append(((k[0]+dx, k[1]+dy), E[k]))
    ## #print "E.....", E_i
    # append all the essential particles to S_final
    for val in E_i:
      S_final.append(val)

    # list of coords with 0 probability
    del_list = []
    # find the probability of distance
    for k in range(len(S_final)):
      ## #print S_final[k][0], my_pos, manhattanDistance(S_final[k][0], my_pos),noisy_dist, gameState.getDistanceProb(noisy_dist,\
      #                  manhattanDistance(S_final[k][0], my_pos))
      weight = S_final[k][1]*gameState.getDistanceProb(noisy_dist,\
                        manhattanDistance(S_final[k][0], my_pos))
      if weight != 0:
        S_final[k] = (S_final[k][0],weight)
      else:
        del_list.append(S_final[k])
    ## #print S_final

    # delete the zero value probabilities
    for val in del_list:
      S_final.remove(val)

    if len(S_final)<1:
        for i in range(num_particles):
          S_final.append((self.en_start[opp_index], 1.0/num_particles))

    prob =[j for i,j in S_orig]
    list_indices = range(len(S_orig))

    chosen_indices = np.random.choice(list_indices,\
                        ((n - len(S_final)) if (n - len(S_final)>0) else 0), replace=True, p=prob)

    for k in chosen_indices:
      if wasMoved:
        choice_list = []
        for dx, dy in [(0,1), (0,-1), (-1,0), (1,0)]:
          if not walls[S_orig[k][0][0]+dx][S_orig[k][0][1]+dy]:
            choice_list.append(((S_orig[k][0][0]+dx, S_orig[k][0][1]+dy),S_orig[k][1]))
        choice = random.choice(choice_list)
        weight = choice[1]*gameState.getDistanceProb(noisy_dist,\
                            manhattanDistance(choice[0], my_pos))
        if weight!=0:
          S_final.append((choice[0],weight))
      else:
        S_final.append(S_orig[k])

    # normalize the weights again

    eps = 0.0
    for k in range(len(S_final)):
      eps = eps + S_final[k][1]
    for k in range(len(S_final)):
      S_final[k] = (S_final[k][0],S_final[k][1]/eps)

    eps = 0.0
    for k in range(len(S_final)):
      eps = eps + S_final[k][1]

    return S_final

##################################################################
# Method to find shortest distance on graph dictinary

  def graph_shortest_dist(self, G, point_a, point_b):
    ## #print point_a, point_b
    if point_a == point_b:
      return 0, None

    s_dist = 1000000
    start = point_a
    end = point_b
    queue = util.Queue()
    queue.push(start)
    counter = 0
    cost = {}
    cost[start] = 0
    while not queue.isEmpty():
      item = queue.pop()
      counter+=1
      if item == end:
        s_dist = cost[item]
        break
      for neighbor in G[item]:
        if cost.setdefault(neighbor, 0) == 0:
          queue.push(neighbor)
          cost[neighbor] = counter
    return s_dist, cost.keys()

#####################################################
# check if b postion can be used as trap for point a

  def b_traps_a(self, a, b, forSelf):
    G = self.G
    A = G[a]
    temp = G[b]
    G[b] = set()
    count = 0

    if forSelf:
      z = self.edge_coords
    else:
      z = self.enemy_edge_coords

    for point in A:
      for point_e in z:
        dist, x = self.graph_shortest_dist(G, point, point_e)
        if  dist < 100000:
          count+=1
          break
    G[b] = temp
    return count<1
#######################################################
# check if point start has only 1 way back to base camp

  def isATrap(self, start, forSelf):
    G = self.G
    temp = G[start]
    G[start] = set()
    count = 0

    if forSelf:
      z = self.edge_coords
    else:
      z = self.enemy_edge_coords

    for point in temp:
      for point_e in z:
        dist, x = self.graph_shortest_dist(G, point, point_e)
        if  dist < 100000:
          count+=1
          break
    G[start] = temp
    return count<2

######################################################
# utility function to find all paths from point_a to LIST of points in point_b
# appends path to all_paths variable

  def find_all_paths_ut(self, G,visited, path, point_a, point_b, all_paths, depth):

    visited[point_a] = True

    # if point_a == (4, 6):
    #   # #print G[point_a]
    #   for a in G[point_a]:
    #     # #print visited[a]

    start = point_a
    path.append(start)

    if depth<0:
      return True

    if start in point_b:
      ## #print path
      ## #print path[-1]
      all_paths.append(list(path))

    for p in G[start]:
      if visited[p] == False:
        depth -= 1
        self.find_all_paths_ut(G,visited, path, p, point_b, all_paths, depth)


    visited[path[-1]] = False
    path.remove(path[-1])

######################################################
# function to find all paths between two points

  def find_all_paths(self, G, point_a, point_b, depth):

    # initialization of values
    visited = {}
    for p in G.keys():
      visited[p] = False
    path = []
    all_paths = []
    depth = 100

    point_B = [point_b,]

    self.find_all_paths_ut(G,visited, path, point_a, point_B, all_paths, depth)

    return all_paths


######################################################
# function to find coordinates in the system for which,
# if the agent wants to return to base camp
# it has go through choke points,

  def find_trap_coords(self,gameState, forSelf, c_time):
    # # #print "################################"
    layout = gameState.data.layout
    walls = layout.walls
    G = copy.deepcopy(self.G)
    traps = {}


    if forSelf:
      pertinent_coords = [(i,j) for i, j in G.keys() \
                  if (self.red and i<walls.width/2) \
                  or (not self.red and i>=walls.width/2)]
    else:
      pertinent_coords = [(i,j) for i, j in G.keys() \
                  if (self.red and i>=walls.width/2) \
                  or (not self.red and i<walls.width/2)]

    for p in G.keys():
      x = set(G[p])
      for item in x:
        if item not in pertinent_coords:
          G[p].remove(item)

    for p in G.keys():
      if p not in pertinent_coords:
        del G[p]

    # # #print len(G.keys()), len(pertinent_coords), walls.height, walls.width

    trap_coords = []
    for point in pertinent_coords:
      if self.isATrap(point, forSelf):
        trap_coords.append(point)
      if time.time() - c_time > 7.0:
        break

    for a in trap_coords:
      for b in trap_coords:
        max_dist = -1
        if self.b_traps_a(a, b, forSelf):
          dist = self.getMazeDistance(a, b)
          if max_dist < dist:
            max_dist = dist
            traps[a] = b
      if time.time() - c_time > 7.0:
            break

    for p in trap_coords:
      if p not in traps.keys():
        traps[p] = p

    ## #print trap_coords
    return traps


##################################################################
# register initial state of the Agent
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    # You can profile your evaluation time by uncommenting these lines
    self.abs_time_taken = 0
    self.max_step_time = 0

    start = time.time()
    #time.sleep(10)
    CaptureAgent.registerInitialState(self, gameState)
    ## #print self.S
    '''
    Your initialization code goes here, if you need any.
    '''
    start1 = time.time()
    #print 'super call', (start1 - start)
    # store history of moves for agent

    self.move_history = util.Queue()
    self.move_repeat_counter = 0
    self.reset_counter = 0

    # high frequency nodes through which enemy agent is more likely to pass
    self.ht_node_list = []

    # coordinated on the boundary with enemy agent
    self.edge_coords = []
    self.enemy_edge_coords = []

    # start position of the current agent
    self.start = gameState.getAgentPosition(self.index)

    # walls in the layout
    walls = gameState.getWalls()

    # width and height of the layout
    width = walls.width
    length = walls.height

    #enemy position history
    self.prev_enemy_pos = dict()
    self.run_counter = 0
    self.was_offensive = False

    # food history
    self.food_eaten = None

    # Graph : store the entire layout in a dictinary
    G = {}

    # Stores the current occupier information for a node in the graph
    G_occupier = {}
    # 0 - Nothing
    # 1 - Food u need to get
    # 2 - Agent A self
    # 3 - Agent B ally
    # 4 - Agent A enemy
    # 5 - Agent B enemy

    # Create the graph
    for i,j in itertools.product(range(width), range(length)):
      if not walls[i][j]:
        if not walls[i-1][j]:
          add_edge(G, (i, j), (i-1, j))
        if not walls[i+1][j]:
          add_edge(G,(i, j), (i+1, j))
        if not walls[i][j-1]:
          add_edge(G,(i, j), (i, j-1))
        if not walls[i][j+1]:
          add_edge(G,(i, j), (i, j+1))
    start2 = time.time()
    #print 'creating graph', (start2 - start1)

    # get the food being defended by the agent
    self.my_food = self.getFoodYouAreDefending(gameState).asList()

    # the value matrix used to initialize value iteration
    self.value_matrix = {}

    self.enemy_pos_history = {}
    self.enemy_pos_history_counter = {}
    # initialize value matrix and boundary coordinates
    for i,j in itertools.product(range(width), range(length)):
      if not walls[i][j]:
        self.value_matrix[(i,j)] = 0
        if (i == (width/2)-1 and not walls[i+1][j]):
          if self.red:
            self.edge_coords.append((i,j))
          else:
            self.enemy_edge_coords.append((i,j))
        elif (i==(width/2) and not walls[i-1][j]):
          if not self.red:
            self.edge_coords.append((i,j))
          else:
            self.enemy_edge_coords.append((i,j))

        if(self.red and i <width/2):
          self.value_matrix[i,j] = 0 #(-(width/2)+i) if (-(width/2)+i) < -3 else 0
        elif(not self.red and i>=width/2):
          self.value_matrix[i,j] = 0 #((width/2)-i)  if ((width/2)-i) < -3 else 0

    ## #print self.edge_coords, self.enemy_edge_coords
    # set food in graph occupier
    for i, j in self.getFood(gameState).asList():
      G_occupier[(i,j)] = 1

    # set agent's position
    G_occupier[self.start] = 2

    if self.index < 2:
      ally_index = self.index+2
    else:
      ally_index = self.index-2

    # set ally agent's position
    ally_pos = gameState.getAgentPosition(ally_index)
    G_occupier[ally_pos]= 3

    min_dist_self = 100000
    min_dist_ally = 100000
    for p in self.getFood(gameState).asList():
      dist_s = self.getMazeDistance(p,self.start)
      dist_a = self.getMazeDistance(p,ally_pos)
      if dist_s < min_dist_self:
        min_dist_self = dist_s
      if dist_a < min_dist_ally:
        min_dist_ally = dist_a

    if min_dist_self>min_dist_ally:
      self.offensive = False
    elif min_dist_self==min_dist_ally:
      if ally_index > self.index:
        self.offensive = False
      else :
        self.offensive = True
    else:
      self.offensive = True

    self.G = G
    self.G_occupier = G_occupier

    # set enemy agent's position
    en1_pos = gameState.getAgentPosition(self.getOpponents(gameState)[0])
    en2_pos = gameState.getAgentPosition(self.getOpponents(gameState)[1])
    self.en_start = {}
    self.en_start[self.getOpponents(gameState)[0]] = en1_pos
    self.en_start[self.getOpponents(gameState)[1]] = en2_pos
    self.prev_enemy_pos[self.getOpponents(gameState)[0]] = None
    self.prev_enemy_pos[self.getOpponents(gameState)[1]] = None
    G_occupier[en1_pos] = 4
    G_occupier[en2_pos] = 5

    ## initialize approx. position particles
    num_particles = 20
    for j in range(num_particles):
      if len(self.S[self.getOpponents(gameState)[0]])<num_particles :
        self.S[self.getOpponents(gameState)[0]].append((en1_pos,1.0/num_particles))
      if len(self.S[self.getOpponents(gameState)[1]])<num_particles :
        self.S[self.getOpponents(gameState)[1]].append((en2_pos,1.0/num_particles))
    self.G = G

    # value matrix for defensive agent
    # Calculate betweenness or high frequency nodes that
    # enemy agent must cross to get to food agent is defending
    betw_edge_list = [random.choice(self.edge_coords),]
    betw_food_list = random.sample(self.my_food, 10)
    vp_betweenness = betweenness(self, G, betw_food_list, [en1_pos,], [en2_pos,], betw_edge_list, None, None, start)

    start2 = time.time()
    #print 'betweenness', (start2 - start1)

    self.G_occupier = G_occupier
    self.vp_betweenness = vp_betweenness

    if not self.offensive:
      self.trap_coords = self.find_trap_coords(gameState, True, start)
    else:
      self.enemy_trap_coords = self.find_trap_coords(gameState, False, start)

    ht_node_list, ht_node_score = self.getHTNodeList(gameState)
    if len(self.ht_node_list)==0:
      self.ht_node_list = ht_node_list
    myPos = gameState.getAgentState(self.index).getPosition()
    if myPos in self.ht_node_list:
        self.ht_node_list.remove(myPos)
    for node in self.ht_node_list:
      self.value_matrix[node] = ht_node_score[node]*100
    self.ht_node_list = ht_node_list
    self.ht_node_score = ht_node_score

    self.defensiveResetMatrix = {}
    for i,j in itertools.product(range(width), range(length)):
      if not walls[i][j]:
        if(self.red and i <width/2):
          self.defensiveResetMatrix[i,j] = -self.getMazeDistance((i,j), self.enemy_edge_coords[0])
        elif(not self.red and i>=width/2):
          self.defensiveResetMatrix[i,j] = -self.getMazeDistance((i,j), self.enemy_edge_coords[0])
    start3 = time.time()
    #print 'coords', (start3 - start2)
    ## #print self.trap_coords
    ## #print self.enemy_trap_coords

    #print 'rego eval time for agent %d: %.4f' % (self.index, time.time() - start)


  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def  update_on_eating_enemy(self, gameState, action):
    num_particles = 20
    S = self.S
    walls = gameState.getWalls()
    width = walls.width
    length =  walls.height
    successor = self.getSuccessor(gameState, action)
    pos = successor.getAgentPosition(self.index)
    for opp_index in self.getOpponents(gameState):
      if gameState.getAgentPosition(opp_index) == pos:
        S[opp_index] = []
        for i in range(num_particles):
          S[opp_index].append((self.en_start[opp_index], 1.0/num_particles))

    for i,j in itertools.product(range(width), range(length)):
      if not walls[i][j]:
        self.reset((i,j),width)

    ## #print S
    return S

  def movePacmanToFood(self, gameState):
        foodList = self.getFood(gameState).asList()
        #print foodList
        idx = 0
        if not self.red:
              idx = len(foodList) - 1
        return self.movePacmanToLoc(gameState, foodList[idx])

  def movePacmanToLoc(self, gameState, loc):
        actions = gameState.getLegalActions(self.index)
        bestDist = 9999
        bestAction = actions[0]
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance( pos2, loc)
            if dist < bestDist:
                bestAction = action
                bestDist = dist
        return bestAction

######################################################################
# Choose Action

  def chooseAction(self, gameState):
    start = time.time()
    if self.offensive == True:
      s  = set(self.prev_10_actions)
      if self.randomise or (len(self.prev_10_actions) == 10 and len(s) < 4  and self.counter < 6):
            self.randomise = True
            self.counter += 1
            self.prev_10_actions.pop(0)
            # x = self.movePacmanToFood(gameState)
            x = random.choice(gameState.getLegalActions(self.index))
      else:
            x =  self.offensiveAgentAction(gameState, start)
      if len(self.prev_10_actions) >= 10:
        self.prev_10_actions.pop(0)
      myPos = gameState.getAgentPosition(self.index)
      newPos = myPos
      if x == Directions.NORTH:
            newPos = (myPos[0], myPos[1] + 1)
      elif x == Directions.SOUTH:
            newPos = (myPos[0], myPos[1] - 1)
      elif x == Directions.WEST:
            newPos = (myPos[0] - 1, myPos[1])
      elif x == Directions.EAST:
            newPos = (myPos[0] + 1, myPos[1])
      self.prev_10_actions.append(newPos)
      if self.counter >= 5:
            self.counter = 0
            self.randomise = False
    else:
      # #print self.index,"defensive", self.red
      x = self.defensiveAgentAction(gameState, start)

    self.abs_time_taken += ( time.time() - start)
    step_time = ( time.time() - start)
    if step_time > self.max_step_time:
      self.max_step_time = step_time
    # #print ("Red " if self.red else "Blue "),"Total Time Taken = ",self.abs_time_taken, "Max Time Taken = ",self.max_step_time

    return x

###################
# Defensive Agent #
###################
  def defensiveAgentAction(self, gameState, c_time):
    start = time.time()
    walls = gameState.getWalls()
    width = walls.width
    length = walls.height
    actions = gameState.getLegalActions(self.index)
    # #print "##########def#######my",("red" if self.red else "blue"),"TeamAgent",self.index,"#################",gameState.getAgentPosition(self.index)
    ## #print "width", width/2
    my_state = gameState.getAgentState(self.index)
    if my_state.scaredTimer > 0:
      self.was_offensive  = True
      return self.offensiveAgentAction(gameState, c_time)
    walls = copy.deepcopy(gameState.getWalls())
    # Enemy Agent's position
    self.S = self.getApproxEnemyAgentPos(gameState)
    ## #print "2nd", self.S
    # if all the value_matrix points are less than eq 0 then RESET

    if time.time() - c_time > 0.8:
      return self.movePacmanToFood(gameState)
    # Assign Negative Values to enemy side:
    for i,j in itertools.product(range(width), range(length)):
        if not walls[i][j]:
          if(self.red and i >=width/2):
            self.value_matrix[i,j] = -7 #(-(width/2)+i) if (-(width/2)+i) < -3 else 0
          elif(not self.red and i<width/2):
            self.value_matrix[i,j] = -7 #((width/2)-i)  if ((width/2)-i) < -3 else 0
          elif (self.red and i<width/2):
                self.value_matrix[i,j] = self.defensiveResetMatrix[(i,j)]
          elif (not self.red and i>=width/2):
            self.value_matrix[i,j] = self.defensiveResetMatrix[(i,j)]
    chk_vm_pos = False
    for key in self.value_matrix:
      if self.value_matrix[key] > 0:
        chk_vm_pos = True
        break
    if not chk_vm_pos or  self.was_offensive:
      self.was_offensive = False
      ht_node_list = self.ht_node_list
      ht_node_score = self.ht_node_score
      if len(self.ht_node_list)==0:
        self.ht_node_list = ht_node_list
      myPos = gameState.getAgentState(self.index).getPosition()
      if myPos in self.ht_node_list:
          self.ht_node_list.remove(myPos)
      for node in self.ht_node_list:
        self.value_matrix[node] = ht_node_score[node]*100

    if time.time() - c_time > 0.8:
      return self.movePacmanToFood(gameState)

    if time.time() - c_time > 0.8:
      return self.movePacmanToFood(gameState)
    # Assign positive values to enemy agents
    for opp_index in self.getOpponents(gameState):
      for apx_pos, w in self.S[opp_index]:
        if(self.red and apx_pos[0]<(width/2)-1) or\
        (not self.red and apx_pos[0]>=(width/2)+1):
          ## #print "enemy3",apx_pos
          if self.value_matrix[apx_pos] < 1000*w:
            self.value_matrix[apx_pos] = 1000*w
    ## #print self.value_matrix
    if time.time() - c_time > 0.8:
      return self.movePacmanToFood(gameState)
    # Assign high positive values to exact enemy agents
    for opp_index in self.getOpponents(gameState):
      enemy_pos = gameState.getAgentPosition(opp_index)
      if enemy_pos!= None and \
        ((self.red and enemy_pos[0]<(width/2))or\
        (not self.red and enemy_pos[0]>=(width/2))):
        ## #print "enemy2",enemy_pos
        self.value_matrix[enemy_pos] = 10000
        if self.prev_enemy_pos[opp_index] != None :
          ## #print "enemy1 removed ###############",self.prev_enemy_pos[opp_index]
          self.value_matrix[self.prev_enemy_pos[opp_index]] = -100000
        self.prev_enemy_pos[opp_index]  = enemy_pos
      elif enemy_pos!= None and \
        ((self.red and enemy_pos[0]>=(width/2))or\
        (not self.red and enemy_pos[0]<(width/2))):
        self.value_matrix[enemy_pos] = -100000

    if time.time() - c_time > 0.8:
      return self.movePacmanToFood(gameState)

    ## #print self.value_matrix
    self.food_eaten = set(self.my_food) - set(self.getFoodYouAreDefending(gameState).asList())
    self.my_food = self.getFoodYouAreDefending(gameState).asList()
    ## #print "this may be error", len(self.food_eaten)
    if len(self.food_eaten)>0:
      ## #print "Found food eater"
      #time.sleep(2)
      self.my_food = self.getFoodYouAreDefending(gameState).asList()
      for i,j in itertools.product(range(width), range(length)):
        if not walls[i][j]:
          self.value_matrix[(i,j)] = 0
          if(self.red and i <width/2):
            self.value_matrix[i,j] = 0 #(-(width/2)+i) if (-(width/2)+i) < -3 else 0
          elif(not self.red and i>=width/2):
            self.value_matrix[i,j] = 0 #((width/2)-i)  if ((width/2)-i) < -3 else 0
      for food in self.food_eaten:
        ## #print "food", food, self.value_matrix[food]
        self.value_matrix[food] = 100000
        if food in self.trap_coords.keys():
          self.value_matrix[self.trap_coords[food]] = 100000
    ## #print self.value_matrix
    if time.time() - c_time > 0.8:
      return self.movePacmanToFood(gameState)
    x = self.get_VI_route(gameState, self.value_matrix, 0, gameState.getAgentPosition(self.index), walls, c_time)

    if x[0]>gameState.getAgentPosition(self.index)[0]:
        action =  Directions.EAST
    elif x[0]<gameState.getAgentPosition(self.index)[0]:
        action =  Directions.WEST
    elif x[1]<gameState.getAgentPosition(self.index)[1]:
        action =  Directions.SOUTH
    elif x[1]>gameState.getAgentPosition(self.index)[1]:
        action =  Directions.NORTH
    else:
        action =  Directions.STOP

    self.S = self.update_on_eating_enemy(gameState, action)
    ## #print "last", x

    self.reset(x, width)

    #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    ## #print self.value_matrix
    return action

  def reset(self,x, width):
    self.value_matrix[x] = 0
    if(self.red and x[0] <width/2):
      self.value_matrix[x] = 0 #-(width/2)+x[0] if (-(width/2)+x[0]) < -3 else 0
    elif(not self.red and x[0]>=width/2):
      self.value_matrix[x] = 0 #(width/2)-x[0]  if ((width/2)-x[0]) < -3 else 0

###################
# Offensive Agent #
###################
  def offensiveAgentAction(self, gameState, c_time):
    """
    Picks among actions randomly.
    """
    ## #print gameState.getAgentPosition(self.index)
    my_pos = gameState.getAgentPosition(self.index)
    for opp_index in self.getOpponents(gameState):
      agent_state = gameState.getAgentState(opp_index)
      agent_pos = gameState.getAgentPosition(opp_index)
      my_state = gameState.getAgentState(self.index)
      if agent_state.isPacman and agent_pos!=None\
        and self.getMazeDistance(agent_pos,my_pos)<2 \
        and my_state.scaredTimer < 1 and agent_state.scaredTimer<1:
        return self.defensiveAgentAction(gameState, c_time)

    actions = gameState.getLegalActions(self.index)
    start = time.time()
    walls = copy.deepcopy(gameState.getWalls())

    width = walls.width
    length = walls.height

    self.warning_counter = 0

    if self.red:
      capsule = gameState.getBlueCapsules()
    else:
      capsule = gameState.getRedCapsules()

    # data-structure to loop through all the important enemy positions
    opp_pos_set = set()
    for opp_index in self.getOpponents(gameState):
      pos = gameState.getAgentPosition(opp_index)
      if pos != None:
        opp_pos_set.add(pos)
    for pos in self.enemy_pos_history.values():
      if pos != None:
        opp_pos_set.add(pos)

    if time.time() - c_time > 0.9:
      return self.movePacmanToFood(gameState)

    # #print "#######off##########my",("red" if self.red else "blue"),"TeamAgent",self.index,"#################"

    G_occupier = self.G_occupier

    actions = gameState.getLegalActions(self.index)
    if self.warning_counter > 0:
      self.warning_counter-=1
      actions.remove('Stop')
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction
    if time.time() - c_time > 0.8:
      return self.movePacmanToFood(gameState)

    for i,j in itertools.product(range(width), range(length)):
      if not walls[i][j]:
        self.value_matrix[(i,j)] = 0
        if(self.red and i <width/2):
          self.value_matrix[(i,j)] = 0 #(-(width/2)+i) if (-(width/2)+i) < -3 else 0
        elif(not self.red and i>=width/2):
          self.value_matrix[(i,j)] = 0 #((width/2)-i)  if ((width/2)-i) < -3 else 0
    if time.time() - c_time > 0.8:
      return self.movePacmanToFood(gameState)

    # Enemy Agent's position
    self.S = self.getApproxEnemyAgentPos(gameState)
    # vp_betweenness = betweenness(self, G, my_food, en1_pos, en2_pos, self.edge_coords, None, None)
    en_scared_count = 0
    # Assign negative values to enemy agents
    for i,j in itertools.product(range(width), range(length)):
        if not walls[i][j]:
          if(self.red and i >=width/2):
            self.value_matrix[i,j] = -7 #(-(width/2)+i) if (-(width/2)+i) < -3 else 0
          elif(not self.red and i<width/2):
            self.value_matrix[i,j] = -7 #((width/2)-i)  if ((width/2)-i) < -3 else 0
          elif (self.red and i<width/2):
            self.value_matrix[i,j] = self.defensiveResetMatrix[(i,j)]
          elif (not self.red and i>=width/2):
            self.value_matrix[i,j] = self.defensiveResetMatrix[(i,j)]
    if time.time() - c_time > 0.8:
      return self.movePacmanToFood(gameState)
    ## assign values to food
    food_val_total = 0
    for i, j in self.getFood(gameState).asList():
        self.value_matrix[i,j] = 99.9
        food_val_total+=99.9
    # Set up walls in case enemy agent's exact position is located:
    for opp_index in self.getOpponents(gameState):
      agent_state = gameState.getAgentState(opp_index)
      ## #print agent_state.scaredTimer, "###scaredTimer"
      self.enemy_pos_history_counter[opp_index] = self.enemy_pos_history_counter.setdefault(opp_index,0)
      if self.enemy_pos_history_counter[opp_index] > 0:
        self.enemy_pos_history_counter[opp_index]-=1
      if self.enemy_pos_history_counter[opp_index] <1:
        self.enemy_pos_history[opp_index] = None
      enemy_pos = gameState.getAgentPosition(opp_index)
      if agent_state.scaredTimer < 1 and\
          (enemy_pos!=None or self.enemy_pos_history[opp_index]!=None):
        my_pos = gameState.getAgentPosition(self.index)
        if enemy_pos != None and my_pos != None:
          dist = self.getMazeDistance(enemy_pos, my_pos)
        ### setup opponent history position (1 prev step)
        if enemy_pos!= None and dist < 8  and\
          ((self.red and enemy_pos[0]>=width/2) or\
          (not self.red and enemy_pos[0]<width/2)):
          self.enemy_pos_history[opp_index]=enemy_pos
          self.enemy_pos_history_counter[opp_index] = 25

        ### value matrix based on enemies
        enemies = set([enemy_pos, self.enemy_pos_history[opp_index]])
        # #print "enemies", enemies
        for enemy_pos in enemies:
          if enemy_pos!= None:
            dist = self.getMazeDistance(enemy_pos, my_pos)
          if enemy_pos!= None and dist < 8 and\
          ((self.red and enemy_pos[0]>=width/2) or\
          (not self.red and enemy_pos[0]<width/2)):
            # #print "error"
            # set up walls
            walls[enemy_pos[0]][enemy_pos[1]] = True
            if (enemy_pos[0],enemy_pos[1]) in self.value_matrix.keys():
              self.value_matrix[(enemy_pos[0],enemy_pos[1])] = -3333
            for dx,dy in [(0,1), (0,-1), (-1,0), (1,0)]:
              if not walls[enemy_pos[0]+dx][enemy_pos[1]+dy]:
                if (enemy_pos[0]+dx,enemy_pos[1]+dy) in self.value_matrix.keys():
                  self.value_matrix[(enemy_pos[0]+dx,enemy_pos[1]+dy)] = -333
              if((self.red and enemy_pos[0]+dx>=width/2) or (not self.red and enemy_pos[0]+dy<width/2)):
                walls[enemy_pos[0]+dx][enemy_pos[1]+dy] = True

            # trap coordinate handling #
            for p in self.enemy_trap_coords:
              if self.value_matrix[p] > -1:
                self.value_matrix[p] = -1

            # set food values
            for p in self.getFood(gameState).asList():
              temp = self.value_matrix[p]
              if self.value_matrix[p] == -1:
                self.value_matrix[p] = 222
              if p  in self.enemy_trap_coords.keys():
                 # trap coordinate handling #
                temp = min(100.9, (self.getMazeDistance(my_pos,enemy_pos) -
                                          2*(self.getMazeDistance(self.enemy_trap_coords[p],p) + 1))*101.9) - 1
                # if temp < self.value_matrix[p]:
                  # # #print "updated food", p, temp
                  # # #print "me-en", self.getMazeDistance(my_pos,enemy_pos), my_pos,enemy_pos
                  # # #print "food-choke", 2*(self.getMazeDistance(self.enemy_trap_coords[p],p) + 1)
              if temp < self.value_matrix[p]:
                self.value_matrix[p] = temp

            # set capsule values
            for cap in capsule:
              if len(capsule)>0 and (agent_state.scaredTimer>2):
                self.value_matrix[cap] = -1
            if dist<5:
              if len(capsule)>0 and (agent_state.scaredTimer<5):
                for i in capsule:
                  self.value_matrix[i] = 999*(gameState.getAgentState(self.index).numCarrying+1)

    if time.time() - c_time > 0.9:
      return self.movePacmanToFood(gameState)
    # edge coordinates's settings
    if en_scared_count < 1:
      my_pos = gameState.getAgentPosition(self.index)
      if ((self.red and my_pos[0]<width/2)or
        (not self.red and my_pos[0]>=width/2)):
        # no longer need to run
        self.run_counter=0
        # setup entry coordinate from home base into enemy territory
        for p in self.edge_coords:
          temp = self.value_matrix[p]
          if len(opp_pos_set) > 0:
            for enemy_pos in opp_pos_set:
              temp_t = min(3333,1099*(manhattanDistance(p,enemy_pos)-3))
              if temp > temp_t:
                temp = temp_t
          else:
            temp = 3333
          if temp < self.value_matrix[p]:
            self.value_matrix[p] = temp
      else:
        # Increase Home base value as agent collect more food:
        for p in self.edge_coords:
          temp = 999999
          if self.value_matrix[p] <=0: self.value_matrix[p]=999999

          if (en_scared_count > 10):
            # no longer need to run
            self.run_counter = 0
            self.value_matrix[p] = 0
          else:
            if len(opp_pos_set)>0:
              for enemy_pos in opp_pos_set:
                temp_t = (gameState.getAgentState(self.index).numCarrying) \
                                      *min(37.3,37.3*(self.getMazeDistance(p,enemy_pos)/ \
                                      self.getMazeDistance(p,my_pos)))
                if temp > temp_t:
                  temp = temp_t
            else:
              temp = 37.3*gameState.getAgentState(self.index).numCarrying
            if temp < self.value_matrix[p]:
              self.value_matrix[p] = temp
      ##############################
      if time.time() - c_time > 0.8:
        return self.movePacmanToFood(gameState)

    foodLeft = len(self.getFood(gameState).asList())
    if foodLeft <= 2:
      self.value_matrix[self.start] = 10000000
      for p in self.edge_coords:
        self.value_matrix[self.start] = 200000
    # #print self.value_matrix
    # #print "scare count",en_scared_count
    ##############################
    x = self.get_VI_route(gameState, self.value_matrix, 0, gameState.getAgentPosition(self.index), walls, c_time)

    if x[0]>gameState.getAgentPosition(self.index)[0]:
        action =  Directions.EAST
    elif x[0]<gameState.getAgentPosition(self.index)[0]:
        action =  Directions.WEST
    elif x[1]<gameState.getAgentPosition(self.index)[1]:
        action =  Directions.SOUTH
    elif x[1]>gameState.getAgentPosition(self.index)[1]:
        action =  Directions.NORTH
    else:
        action =  Directions.STOP
    self.S = self.update_on_eating_enemy(gameState, action)
    #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    # # #print action
    return action
  ##########################################################
  # HT node list
  def getHTNodeList(self,gameState):

    walls = gameState.getWalls()
    width = walls.width
    num_ht_nodes = (int)(0.5 * width)
    ht_node_score = {}
    en1_pos_m = {}
    for p,w in self.S[self.getOpponents(gameState)[0]]:
      en1_pos_m[p] = en1_pos_m.setdefault(p,0)+w

    en2_pos_m = {}
    for p,w in self.S[self.getOpponents(gameState)[1]]:
      en2_pos_m[p] = en2_pos_m.setdefault(p,0)+w
    # # #print en1_pos_m, en2_pos_m
    en1_pos = sorted(en1_pos_m, key=en1_pos_m.__getitem__, reverse = True)
    en2_pos = sorted(en2_pos_m, key=en2_pos_m.__getitem__, reverse = True)
    # # #print en1_pos, en1_pos_m[en1_pos[0]], en2_pos, en2_pos_m[en2_pos[0]]

    #self.vp_betweenness = betweenness(self, self.G, self.my_food, en1_pos[:1], en2_pos[:1], self.edge_coords, None, None)

    for k in self.vp_betweenness:
      if ((self.red and k[0] < (width/2)) or \
              (not self.red and k[0] >= (width/2))):
        ht_node_score[k] = self.vp_betweenness[k]

    ht_node_list = sorted(ht_node_score, key=ht_node_score.__getitem__, reverse = True)[:num_ht_nodes]
    return ht_node_list, ht_node_score
####################
# Utility Function #
####################
#################################################
# adds an edge to a graph dictionary object

def add_edge(graph, u, v):
  graph[u] = graph.setdefault(u,set())
  graph[u].add(v)
  graph[v] = graph.setdefault(v,set())
  graph[v].add(u)

################################################
# find shortest path between two points
# returns a list of points from source to destination

def find_shortest_paths(gameState, graph, point_A, point_B):

  source = [point_A,]
  ## #print source, point_A, point_B
  #time.sleep(4)
  path = [source,]

  while source[0]!= point_B:
    min_dist = 100000
    min_point = []
    for neighbor in graph[source[0]]:
      dist = gameState.getMazeDistance(neighbor, point_B)
      ## #print dist, neighbor
      if dist < min_dist:
        min_dist = dist
        min_point = [neighbor,]
      elif dist==min_dist:
        min_point.append(neighbor)
        ## #print "more paths"
    ## #print min_point

    source = min_point
    for point in source[1:]:
      for part_path in find_shortest_paths(gameState, graph, point, point_B):
        path.append(path[0]+part_path)
        ## #print "herere bad !!"
        #time.sleep(10)
    path[0].append(source[0])
  ## #print "STOP", path
  return path


################################################
# Calculate betweenness to determine the highest traffic nodes
# between enemy agents and food the agents are defending
# returns betweenness values for all points in a dict

def betweenness(gameState, graph, food_points, enemy1, enemy2,\
        entry_points, weights_e1, weights_e2, c_time):
  score = {}
  total_count = 0

  for source, dest in itertools.product(food_points, enemy1):
    total_count+=1
    score_temp = calc_betweenness(gameState, graph, source, dest)
    if time.time() - c_time > 3.5:
          break
    for k, v in score_temp.iteritems(): score[k] = score.setdefault(k,0)+v

  for source, dest in itertools.product(food_points, enemy2):
    total_count+=1
    score_temp = calc_betweenness(gameState, graph, source, dest)
    if time.time() - c_time > 3.5:
          break
    for k, v in score_temp.iteritems(): score[k] = score.setdefault(k,0)+v
  ## #print total_count
  score = {k: v / total_count for k, v in score.iteritems()}
  ## #print score
  return score

###########################################
# util function for calculating betweenness
def calc_betweenness(gameState, graph, point_a, point_b):
  ## #print "points", point_a, point_b
  paths = find_shortest_paths(gameState, graph, point_a, point_b)
  total_paths = (float)(len(paths))
  score = {}
  for path in paths:
      for point in path:
          score[point] = score.setdefault(point, 0) + 1

  score = {k: v / total_paths for k, v in score.iteritems()}
  ## #print "score :", score
  return score
