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

from game import Agent

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

        oldFood = currentGameState.getFood()

        if successorGameState.isWin():
          return 999999
        scared = 0.0
        ghosts = 0.0
        food = 0.0

        for i, g in enumerate(newGhostStates):
          dist = manhattanDistance(newPos, g.getPosition())
          val = dist - newScaredTimes[i]
          if newScaredTimes[i] != 0:
          	scared += (1.0 / val)
          if dist == 0:
            dist = .0000001
          ghosts += (1.0 - (1.0 / dist))

        for i, f in enumerate(newFood.asList()):
          dist = manhattanDistance(newPos, f)
          food += (1.0 / dist)

        fooddiff = len(oldFood.asList()) - len(newFood.asList())
        glen = float(len(newGhostStates))
        flen = float(len(newFood.asList()))

        return 4 * scared / glen + 1.8 * ghosts / glen + 3 * food / flen + fooddiff
        #return successorGameState.getScore()

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

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        def helpAction(gameState, index, depth, start):
          if gameState.isWin():
            return self.evaluationFunction(gameState)
          if gameState.isLose():
            return self.evaluationFunction(gameState)
          if depth == 0:
            return self.evaluationFunction(gameState)
          else:
            # Collect legal moves and successor states
            legalMoves = gameState.getLegalActions(index)

            #increment depth level
            if (index + 1) % gameState.getNumAgents() == start:
              depth -= 1
            nexti = (index + 1) % gameState.getNumAgents()
            # Choose one of the best actions
            scores = [helpAction(gameState.generateSuccessor(index, action), nexti, depth, start) for action in legalMoves]
            if index == 0:
              bestScore = max(scores)
            else:
              bestScore = min(scores)

            return bestScore

        scores = [helpAction(gameState.generateSuccessor(self.index, action), self.index + 1, self.depth, self.index) for action in legalMoves]
        if self.index == 0:
          bestScore = max(scores)
        else:
          bestScore = min(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        alpha = float("-inf")
        beta = float("inf")
        def helpAction(gameState, index, depth, start, a, b):
          if gameState.isWin():
            return self.evaluationFunction(gameState)
          if gameState.isLose():
            return self.evaluationFunction(gameState)

          if depth == 0:
            return self.evaluationFunction(gameState)
          else:
            # Collect legal moves and successor states
            legalMoves = gameState.getLegalActions(index)

            #increment depth level
            if (index + 1) % gameState.getNumAgents() == start:
              depth -= 1
            nexti = (index + 1) % gameState.getNumAgents()

            # Choose one of the best actions
            if index == 0:
              best = float("-inf")
              for i, action in enumerate(legalMoves):
                value = helpAction(gameState.generateSuccessor(index, action), nexti, depth, start, a, b) 
                best = max(best, value)
                if best > b:
                  return best
                a = max(a, best)
            else:
              best = float("inf")
              for i, action in enumerate(legalMoves):
                value = helpAction(gameState.generateSuccessor(index, action), nexti, depth, start, a, b) 
                best = min(best, value)
                if best < a:
                  return best
                b = min(b, best)

            return best

        scores = []
        if self.index == 0:
          for i, action in enumerate(legalMoves):
            state = gameState.generateSuccessor(self.index, action)
            value = helpAction(state, self.index + 1, self.depth, self.index, alpha, beta) 
            alpha = max(alpha, value)
            if alpha > beta:
              return action
            scores.append(value)
          bestScore = alpha
        else:
          for i, action in enumerate(legalMoves):
            state = gameState.generateSuccessor(self.index, action)
            value = helpAction(state, self.index + 1, self.depth, self.index, alpha, beta) 
            beta = min(beta, value)
            if beta < alpha:
              return action
            scores.append(value)
            bestScore = beta

        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

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
        if self.index != 0:
          return random.choice(legalMoves)

        def helpExpect(gameState, index, depth, start):
          if gameState.isWin():
            return self.evaluationFunction(gameState)
          if gameState.isLose():
            return self.evaluationFunction(gameState)

          legalMoves = gameState.getLegalActions(index)

          if depth == 0:
            return self.evaluationFunction(gameState)
          elif index != 0:
            #increment depth level
            if (index + 1) % gameState.getNumAgents() == start:
              depth -= 1
            nexti = (index + 1) % gameState.getNumAgents()

            #take average score of moves
            total = 0.0
            for action in legalMoves:
              state = gameState.generateSuccessor(index, action)
              total += float (helpExpect(state, nexti, depth, start))
            return total / float(len(legalMoves))
          else:
            #increment depth level
            if (index + 1) % gameState.getNumAgents() == start:
              depth -= 1
            nexti = (index + 1) % gameState.getNumAgents()

            scores = [helpExpect(gameState.generateSuccessor(index, action), nexti, depth, start) for action in legalMoves]
            return max(scores)

        legalMoves = gameState.getLegalActions(self.index)
        scores = [helpExpect(gameState.generateSuccessor(self.index, action), self.index + 1, self.depth, self.index) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        # Pick randomly among the best

        return legalMoves[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsules = successorGameState.getCapsules()
    if successorGameState.isWin():
      return 999999
    scared = 0.0
    ghosts = 0.0
    food = 0.0
    capsule = 0.0

    for i, g in enumerate(newGhostStates):
      dist = manhattanDistance(newPos, g.getPosition())
      val = dist - newScaredTimes[i]
      if newScaredTimes[i] != 0:
        scared += (1.0 / val)
      if dist == 0:
        dist = .0000001
      ghosts += (1.0 - (1.0 / dist))

    for i, f in enumerate(newFood.asList()):
      dist = manhattanDistance(newPos, f)
      if dist < 3:
        food += dist
      else:
        food += (1.0 / dist)
    for c in capsules:
      dist = manhattanDistance(newPos, c)
      if dist == 0:
        capsule += dist * 4
      else:
        capsule += (1.0 / dist)

    glen = float(len(newGhostStates))
    flen = float(len(newFood.asList()))
    clen = float(len(capsules))

    scared = 4 * scared / glen
    ghosts = 1.3 * ghosts / glen
    food = 3 * food / flen
    if clen == 0:
      clen = .000001
    else:
      capsule = 8 * capsule / clen

    return scared + ghosts + food + capsule + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

