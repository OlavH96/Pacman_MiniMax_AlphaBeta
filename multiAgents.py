# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from time import sleep

from game import Agent

NEGATIVE_INFINITY = float('-inf')
POSITIVE_INFINITY = float('inf')
PACMAN = 0
INITIAL_DEPTH = 0

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    # Extra class for keeping track of actions and their scores, basically the same as scoring a state, but we want to find the action that leads to a state.
    # this avoid having 2 values or tuples to pass around.
    class ActionScore:
        def __init__(self, action, score):
            self.action = action
            self.score = score
        def __str__(self):
            return "action="+str(self.action)+", score="+str(self.score)

    # Check if state is terminal
    def isTerminal(self, state):
        return state.isWin() or state.isLose()
    # Find largest possible score you can get if opponent plays optimally (used for pacman)
    def max_value(self, state, agent, depth):

        max_score = self.ActionScore(Directions.STOP, NEGATIVE_INFINITY) # Initial state
        actions = state.getLegalActions(agent) # get the legal actions
        for action in actions:
            next_state = state.generateSuccessor(agent, action) # Generate next state from that action
            next_depth = depth+1 # increment depth

            next_value = self.evaluate_state(next_state, PACMAN+1, next_depth) # evaluate next state, max_value is always used for pacman, so we dont need to calculate the next agent
            next_ActionScore = self.ActionScore(action, next_value) # create struct to hold value, so we do not have to deal with indices and two arrays

            max_score = max([max_score, next_ActionScore], key = lambda e : e.score) # check if new action has higher score

        return max_score

    # Find smallest possible score you can get if opponent plays optimally (used for ghosts)
    def min_value(self, state, agent, depth):

        min_score = self.ActionScore(Directions.STOP, POSITIVE_INFINITY) # initial state
        actions = state.getLegalActions(agent) # get legal moves
        for action in actions: # check every action
            next_state = state.generateSuccessor(agent, action) # generate next state
            next_depth = depth+1 # increment depth
            next_agent = (agent+1) % self.number_of_agents # Find the next agent by adding one and doing a mod operation. If there are 3 agents this will go 0 -> 1 -> 2 -> 0, etc

            next_value = self.evaluate_state(next_state, next_agent, next_depth) # evaluate state
            next_ActionScore = self.ActionScore(action, next_value) # create struct

            min_score = min([min_score, next_ActionScore], key = lambda e : e.score) # check if new action has lower score, lower is better in this case (worse for pacman)
        return min_score

    # This could just be in min_ and max_value, but I found this more readable
    def evaluate_state(self, state, agent, depth): # Evaluates a state
        if self.isTerminal(state) or depth == self.depth_with_agents: # if we reached the end (win/loss), or recursion depth, return the score for the state.
            return self.evaluationFunction(state)
        # Since we have more than 2 agents, we can not simply use the algorithm in the lecture slides, we have to check which
        # agent is next, so I added this method to simplify this.
        # If a max node calls this function, it is essentially the same as calling min_value directly
        # but if a min node calls it, it can call both min_ or max_value, depending on the next agent.
        # If we had more than two maximizing agents, we could modify the below if sentence to allow this. For example if you are playing a game with 6 players with 2 player on your team, and 4 players on the other team.
        if (agent == PACMAN): # If the agent is pacman, find the max possible score value
            return self.max_value(state, agent, depth).score
        else: # if the agent is a not pacman, find the min possible score value (worst for pacman), this means we assume the ghosts play optimally.
            return self.min_value(state, agent, depth).score

    def getAction(self, state):

        self.number_of_agents = state.getNumAgents() # store the number of agents
        self.depth_with_agents = self.depth * self.number_of_agents

        action_with_score = self.max_value(state, PACMAN, INITIAL_DEPTH) # Find the action with the max value from the initial state for pacman.
        #print action_with_score
        return action_with_score.action # Get the action, since that is what we want to determine

class AlphaBetaAgent(MultiAgentSearchAgent):

    # Extra class for keeping track of actions and their scores, basically the same as scoring a state, but we want to find the action that leads to a state.
    class ActionScore:
        def __init__(self, action, score):
            self.action = action
            self.score = score
        def __str__(self):
            return "action="+str(self.action)+", score="+str(self.score)

    # Check if state is terminal
    def isTerminal(self, state):
        return state.isWin() or state.isLose()
    # Find largest possible score you can get if opponent plays optimally (used for pacman)
    def max_value(self, state, agent, depth, alpha, beta):

        output = self.ActionScore(Directions.STOP, NEGATIVE_INFINITY) # Initial state
        actions = state.getLegalActions(agent) # Get the legal actions
        for action in actions:
            next_state = state.generateSuccessor(agent, action) # Generate next state from that action
            next_depth = depth+1 # increment depth
            # in this case the next agent is always agent #1, so we dont have to calculate the next agent like we do in min_value
            next_value = self.evaluate_state(next_state, PACMAN+1, next_depth, alpha, beta) # evaluate next state
            next_ActionScore = self.ActionScore(action, next_value) # create struct to hold value, so we do not have to deal with indices and two arrays

            output = max([output, next_ActionScore], key = lambda e : e.score) # check if new action has higher score
            # if you use <= like in the lectures, it does not work (tests fail), however < works (?)
            if beta < output.score:
                return output
            alpha = max(alpha, output.score) # Only max nodes can modify alpha
        return output

    # Find smallest possible score you can get if opponent plays optimally (used for ghosts)
    def min_value(self, state, agent, depth, alpha, beta):

        output = self.ActionScore(Directions.STOP, POSITIVE_INFINITY) # initial state
        actions = state.getLegalActions(agent) # get legal moves
        for action in actions: # check every action
            next_state = state.generateSuccessor(agent, action) # generate next state
            next_depth = depth+1 # increment depth
            next_agent = (agent+1) % self.number_of_agents # Find the next agent by adding one and doing a mod operation. If there are 3 agents this will go 0 -> 1 -> 2 -> 0, etc

            next_value = self.evaluate_state(next_state, next_agent, next_depth, alpha, beta) # evaluate state
            next_ActionScore = self.ActionScore(action, next_value) # create struct

            output = min([output, next_ActionScore], key = lambda e : e.score) # check if new action has lower score, lower is better in this case (worse for pacman)
            # if you use >= like in the lectures, it does not work (tests fail), however > works (?)
            if alpha > output.score:
                return output
            beta = min(beta, output.score) # only min nodes can modify beta
        return output

    def evaluate_state(self, state, agent, depth, alpha, beta): # Evaluates a state
        if self.isTerminal(state) or depth == self.depth_with_agents: # if we reached the end (win/loss), or recursion depth, return the score for the state.
            return self.evaluationFunction(state)

        if (agent == PACMAN): # If the agent is pacman, find the max possible score value
            return self.max_value(state, agent, depth, alpha, beta).score
        else: # if the agent is not pacman, find the min possible score value (worst for pacman), this means we assume the ghosts play optimally.
            return self.min_value(state, agent, depth, alpha, beta).score

    def getAction(self, state):
        # This is the same algorithm as the first task, except I've tagged on alpha and beta values.
        # See Minimax class for more detailed comments on this class.
        alpha = NEGATIVE_INFINITY # Start alpha at min possible value
        beta = POSITIVE_INFINITY # Start beta at max possible value

        self.number_of_agents = state.getNumAgents() # store the number of agents
        self.depth_with_agents = self.depth * self.number_of_agents

        action_with_score = self.max_value(state, PACMAN, INITIAL_DEPTH, alpha, beta) # Find the action with the max value from the initial state for pacman.
        #print action_with_score
        return action_with_score.action # Get the action, since that is what we want to determine

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
