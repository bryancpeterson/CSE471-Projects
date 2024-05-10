# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Recommended to use a stack? Init
    stack = util.Stack()

    # Push the start state and an empty list of actions to the stack
    stack.push((problem.getStartState(), []))

    # Initialize a set to keep track of visited states
    visited = set()

    while not stack.isEmpty():
        # Pop the current state and the list of actions
        current_state, actions = stack.pop()

        # Check if the current state is the goal state
        if problem.isGoalState(current_state):
            return actions

        # Mark the current state as visited
        visited.add(current_state)

        # Get the successors of the current state
        successors = problem.getSuccessors(current_state)

        for successor in successors:
            next_state, action, _ = successor

            # Check if the next state has not been visited
            if next_state not in visited:
                # Push the next state and the updated list of actions to the stack
                stack.push((next_state, actions + [action]))

    # If no solution is found, return an empty list
    return []


    #util.raiseNotDefined() never gets hit

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Initialize a queue for BFS
    queue = util.Queue()

    # Push the start state and an empty list of actions to the queue
    queue.push((problem.getStartState(), []))

    # Initialize a set to keep track of visited states
    visited = set()

    while not queue.isEmpty():
        # Pop the current state and the list of actions
        current_state, actions = queue.pop()

        # Check if the current state has already been visited
        if current_state in visited:
            continue

        # Check if the current state is the goal state
        if problem.isGoalState(current_state):
            return actions

        # Mark the current state as visited
        visited.add(current_state)

        # Get the successors of the current state
        successors = problem.getSuccessors(current_state)

        for successor in successors:
            next_state, action, _ = successor

            # Check if the next state has not been visited
            if next_state not in visited:
                # Push the next state and the updated list of actions to the queue
                queue.push((next_state, actions + [action]))

    # If no solution is found, return an empty list
    return []
    
    #util.raiseNotDefined() never gets hit

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Initialize a priority queue for UCS
    priority_queue = util.PriorityQueue()

    # Push the start state, an empty list of actions, and the initial cost (0) to the priority queue
    priority_queue.push((problem.getStartState(), [], 0), 0)

    # Initialize a set to keep track of visited states
    visited = set()

    while not priority_queue.isEmpty():
        # Pop the current state, list of actions, and cost
        current_state, actions, cost = priority_queue.pop()

        # Check if the current state has already been visited
        if current_state in visited:
            continue

        # Check if the current state is the goal state
        if problem.isGoalState(current_state):
            return actions

        # Mark the current state as visited
        visited.add(current_state)

        # Get the successors of the current state
        successors = problem.getSuccessors(current_state)

        for successor in successors:
            next_state, action, step_cost = successor

            # Check if the next state has not been visited
            if next_state not in visited:
                # Push the next state, updated list of actions, and updated cost to the priority queue
                priority_queue.push((next_state, actions + [action], cost + step_cost), cost + step_cost)

    # If no solution is found, return an empty list
    return []


    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Initialize a priority queue for A* search
    priority_queue = util.PriorityQueue()

    # Push the start state, an empty list of actions, and the initial cost (0 + heuristic) to the priority queue
    start_state = problem.getStartState()
    priority_queue.push((start_state, [], 0), heuristic(start_state, problem))

    # Initialize a set to keep track of visited states
    visited = set()

    while not priority_queue.isEmpty():
        current_state, actions, cost = priority_queue.pop()

        if current_state in visited:
            continue

        if problem.isGoalState(current_state):
            return actions
        visited.add(current_state)

        successors = problem.getSuccessors(current_state)

        for successor in successors:
            next_state, action, step_cost = successor

            if next_state not in visited:
                new_cost = cost + step_cost

                # Calculate the heuristic estimate to the goal from the neighbor (h-value)
                h_value = heuristic(next_state, problem)

                # Calculate the total estimated cost (f-value = g-value + h-value)
                f_value = new_cost + h_value

                # Push the next state, updated list of actions, and updated cost to the priority queue
                priority_queue.push((next_state, actions + [action], new_cost), f_value)

    # If no solution is found, return an empty list
    return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch