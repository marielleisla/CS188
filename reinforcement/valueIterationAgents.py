# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        count = 0

        # Iterate for self.iterations times
        while count < self.iterations:
            new = self.values.copy()
            # Loop for all states 
            for state in self.mdp.getStates():
                q_values = util.Counter()
                # Compute Q-values
                if len(self.mdp.getPossibleActions(state)) == 0:
                    new[state] = 0
                else:
                    for action in self.mdp.getPossibleActions(state):
                        q_values[action] = self.computeQValueFromValues(state, action)
                    # Take the value of action that has the highest Q-value
                    best = q_values.argMax()
                    new[state] = q_values[best]
            # Increment counter & copy for next iteration
            count += 1
            self.values = new


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        transitions = util.Counter()
        # iterate through transition states
        for pair in self.mdp.getTransitionStatesAndProbs(state, action):
            (nextState, prob) = pair
            reward = self.mdp.getReward(state, action, nextState)
            transitions[pair] = prob * (reward + self.discount * self.values[nextState])
        # sum up values
        return transitions.totalCount()


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        q_values = util.Counter()
        # Compute Q-values
        if len(self.mdp.getPossibleActions(state)) == 0:
            return None
        for action in self.mdp.getPossibleActions(state):
            q_values[action] = self.computeQValueFromValues(state, action)
        # Take the action that has the highest Q-value
        return q_values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        count = 0

        # Iterate for self.iterations times
        while count < self.iterations:
            new = self.values.copy()
            # Loop for all states 
            for state in self.mdp.getStates():
                if count >= self.iterations:
                    return
                if self.mdp.isTerminal(state):
                    count += 1
                    continue

                q_values = util.Counter()
                # Compute Q-values
                if len(self.mdp.getPossibleActions(state)) == 0:
                    self.values[state] = 0
                else:
                    for action in self.mdp.getPossibleActions(state):
                        q_values[action] = self.computeQValueFromValues(state, action)
                    # Take the value of action that has the highest Q-value
                    best = q_values.argMax()
                    self.values[state] = q_values[best]
                # Increment counter
                count += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        #Compute predecessors of all states
        predecessors = {}
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for pair in self.mdp.getTransitionStatesAndProbs(state, action):
                    (nextState, prob) = pair
                    if prob > 0:
                        #predecessors[nextState].append(state)
                        if nextState in predecessors:
                            predecessors[nextState] += [state]
                        else:
                            predecessors[nextState] = [state]

        pq = util.PriorityQueue()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            q_values = util.Counter()
            for action in self.mdp.getPossibleActions(state):
                        q_values[action] = self.computeQValueFromValues(state, action)

            # Take the value of action that has the highest Q-value
            best = q_values.argMax()
            best_val = q_values[best]
            diff = abs(self.values[state] - best_val)

            #push into queue w/ priority -diff
            pq.push(state, -diff)

        count = 0
        while count < self.iterations:
            if pq.isEmpty():
                return

            state = pq.pop()

            if not self.mdp.isTerminal(state):
                q_values = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                        q_values[action] = self.computeQValueFromValues(state, action)

                # Take the value of action that has the highest Q-value
                best = q_values.argMax()
                self.values[state] = q_values[best]

            for p in predecessors[state]:
                q_values = util.Counter()
                for action in self.mdp.getPossibleActions(p):
                        q_values[action] = self.computeQValueFromValues(p, action)

                # Take the value of action that has the highest Q-value
                best = q_values.argMax()
                best_val = q_values[best]
                diff = abs(self.values[p] - best_val)

                if diff > self.theta:
                    pq.update(p, -diff)
            count += 1
