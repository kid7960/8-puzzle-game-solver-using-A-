from copy import deepcopy
import numpy as np
import time


"""
This function takes the current states and gives the best path towards the goal state
"""
def bestSolution(state):
    bestSol = np.array([], int).reshape(-1, 9)
    count = len(state) - 1
    while count != -1:
        bestSol = np.insert(bestSol, 0, state[count]['puzzle'], 0)
        count = (state[count]['parent'])
    return bestSol.reshape(-1, 3, 3)


"""
This function calculates number of misplaced tiles in the current state from the goal state
"""
def misplaced_tiles(puzzle,goal):
    mscost = np.sum(puzzle != goal) - 1
    return mscost if mscost > 0 else 0


"""
This function identifies the coordinates of each tile in the states
"""
def coordinates(puzzle):
    pos = np.array(range(9))
    for p, q in enumerate(puzzle):
        pos[q] = p
    return pos


"""
This function solves the 8-puzzle problem using the misplaced tile heuristics function
"""
def solve_misplaced(puzzle, goal):
    steps = np.array([('up', [0, 1, 2], -3), ('down', [6, 7, 8], 3), ('left', [0, 3, 6], -1), ('right', [2, 5, 8], 1)],
                     dtype=[('move', str, 1), ('position', list), ('head', int)])

    dtstate = [('puzzle', list), ('parent', int), ('gn', int), ('hn', int)]

    # initializing the parent, gn and manhattanDistance
    costGoal = coordinates(goal)
    parent = -1
    gn = 0
    hn = misplaced_tiles(coordinates(puzzle), costGoal)
    state = np.array([(puzzle, parent, gn, hn)], dtstate)

    # Initialize priority queues with position as keys and fn as value.
    dtpriority = [('position', int), ('fn', int)]
    priority = np.array([(0, hn)], dtpriority)

    while 1:
        # sort priority queue using merge sort,
        # the first element is picked, remove from queue what we are currently exploring
        priority = np.sort(priority, kind='mergesort', order=['fn', 'position'])
        position, fn = priority[0]
        priority = np.delete(priority, 0, 0)

        puzzle, parent, gn, hn = state[position]
        puzzle = np.array(puzzle)

        # 0 denotes the blank tile
        blank = int(np.where(puzzle == 0)[0])

        # Increase cost g(n) by 1
        gn = gn + 1
        c = 1
        start_time = time.time()

        for s in steps:
            c = c + 1
            if blank not in s['position']:
                # initialize new state as copy of current
                openStates = deepcopy(puzzle)
                openStates[blank], openStates[blank + s['head']] = openStates[blank + s['head']], openStates[blank]

                # Check if the node has been previously explored or not
                if ~(np.all(list(state['puzzle']) == openStates, 1)).any():
                    end_time = time.time()
                    if (end_time - start_time) > 2:
                        break

                    hn = misplaced_tiles(coordinates(openStates), costGoal)

                    # add new state in the list
                    q = np.array([(openStates, position, gn, hn)], dtstate)
                    state = np.append(state, q, 0)

                    # f(n) is the sum of cost reaching current node and the cost of reaching the goal state
                    # from current node
                    fn = gn + hn

                    q = np.array([(len(state) - 1, fn)], dtpriority)
                    priority = np.append(priority, q, 0)

                    # Checking if the node in openStates are matching the goal state.
                    if np.array_equal(openStates, goal):
                        return state, len(priority)

    return state, len(priority)


puzzle = [5, 0, 8, 4, 2, 1, 7, 3, 6]
goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]

state, visited = solve_misplaced(puzzle, goal)
bestpath = bestSolution(state)
print(str(bestpath))
totalmoves = len(bestpath) - 1
print('Steps to reach goal:', totalmoves)
