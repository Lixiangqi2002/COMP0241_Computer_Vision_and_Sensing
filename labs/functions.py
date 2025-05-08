import numpy as np
# the goal of this routine is to return the minimum cost dynamic programming
# solution given a set of unary and pairwise costs

def dynamicProgram_efficient(unaryCosts, pairwiseCosts):

    # count number of positions (i.e. pixels in the scanline), and nodes at each
    # position (i.e. the number of distinct possible disparities at each position)
    nNodesPerPosition = len(unaryCosts)
    nPosition = len(unaryCosts[0])

    minimumCost = np.zeros([nNodesPerPosition, nPosition])
    parents = np.zeros([nNodesPerPosition, nPosition])

    # FORWARD PASS
    minimumCost[:, 0] = unaryCosts[:, 0]

    for cPosition in range(1, nPosition):
        # Vectorized computation for possible path costs
        costs = minimumCost[:, cPosition - 1][:, np.newaxis] + pairwiseCosts
        minCost = np.min(costs, axis=0)  # Minimum cost for each node
        ind = np.argmin(costs, axis=0)  # Index of the minimum cost

        # Update minimum cost and parent matrices
        minimumCost[:, cPosition] = minCost + unaryCosts[:, cPosition]
        parents[:, cPosition] = ind

    # BACKWARD PASS
    bestPath = np.zeros(nPosition, dtype=int)
    bestPath[-1] = np.argmin(minimumCost[:, -1])

    for cPosition in range(nPosition - 2, -1, -1):
        bestPath[cPosition] = parents[bestPath[cPosition + 1], cPosition + 1]

    return bestPath


def dynamicProgram(unaryCosts, pairwiseCosts):

    # count number of positions (i.e. pixels in the scanline), and nodes at each
    # position (i.e. the number of distinct possible disparities at each position)
    nNodesPerPosition = len(unaryCosts)
    nPosition = len(unaryCosts[0])

    # define minimum cost matrix - each element will eventually contain
    # the minimum cost to reach this node from the left hand side.
    # We will update it as we move from left to right
    minimumCost = np.zeros([nNodesPerPosition, nPosition])

    # define parent matrix - each element will contain the (vertical) index of
    # the node that preceded it on the path.  Since the first column has no
    # parents, we will leave it set to zeros.
    parents = np.zeros([nNodesPerPosition, nPosition])

    # FORWARD PASS

    # TODO:  fill in first column of minimum cost matrix
    minimumCost[:, 0] = unaryCosts[:, 0]

    # Now run through each position (column)
    for cPosition in range(1,nPosition):
        # run through each node (element of column)
        for cNode in range(nNodesPerPosition):
            # now we find the costs of all paths from the previous column to this node
            possPathCosts = np.zeros([nNodesPerPosition,1])
            for cPrevNode in range(nNodesPerPosition):
                # TODO  - fill in elements of possPathCosts
                possPathCosts[cPrevNode] = (
                        minimumCost[cPrevNode, cPosition - 1] +
                        pairwiseCosts[cPrevNode, cNode]
                )
            # TODO - find the minimum of the possible paths 
            minCost = np.min(possPathCosts)
            ind = np.argmin(possPathCosts)

            # Assertion to check that there is only one minimum cost.
            # assert(len(np.where(possPathCosts == minCost)[0]) == 1)

            # TODO - store the minimum cost in the minimumCost matrix
            # minimumCost[, ] = 
            minimumCost[cNode, cPosition] = minCost + unaryCosts[cNode, cPosition]
            parents[cNode, cPosition] = ind

            # TODO - store the parent index in the parents matrix
            # parents[, ] = 


    # BACKWARD PASS
    # Initialize the best path
    bestPath = np.zeros(nPosition, dtype=int)
    #TODO  - find the index of the overall minimum cost from the last column and put this
    #into the last entry of best path
    # Find the minimum cost in the last column
    minInd = np.argmin(minimumCost[:, -1])
    bestPath[-1] = minInd

    # TODO - find the parent of the node you just found
    # Trace back the best path
    for cPosition in range(nPosition - 2, -1, -1):
        bestPath[cPosition] = parents[bestPath[cPosition + 1], cPosition + 1]

    return bestPath


def dynamicProgramVec(unaryCosts, pairwiseCosts):
    
    # same preprocessing code
    
    # count number of positions (i.e. pixels in the scanline), and nodes at each
    # position (i.e. the number of distinct possible disparities at each position)
    nNodesPerPosition = len(unaryCosts)
    nPosition = len(unaryCosts[0])

    # define minimum cost matrix - each element will eventually contain
    # the minimum cost to reach this node from the left hand side.
    # We will update it as we move from left to right
    minimumCost = np.zeros([nNodesPerPosition, nPosition])
    parents = np.zeros([nNodesPerPosition, nPosition], dtype=int)

    # FORWARD PASS
    # Initialize first column of minimum cost matrix
    minimumCost[:, 0] = unaryCosts[:, 0]


    # TODO: fill this function in. (hint use tiling and perform calculations columnwise with matricies)
    # Iterate over positions
    for cPosition in range(1, nPosition):
        # Compute possible path costs from previous column (vectorized)
        possPathCosts = (
                minimumCost[:, cPosition - 1].reshape(-1, 1) +
                pairwiseCosts
        )
        # Find the minimum cost and its index for each node
        minCosts = np.min(possPathCosts, axis=0)
        indices = np.argmin(possPathCosts, axis=0)

        # Store the minimum cost and parent index
        minimumCost[:, cPosition] = minCosts + unaryCosts[:, cPosition]
        parents[:, cPosition] = indices

    # BACKWARD PASS
    # Initialize the best path
    bestPath = np.zeros(nPosition, dtype=int)

    # Find the minimum cost in the last column
    minInd = np.argmin(minimumCost[:, -1])
    bestPath[-1] = minInd

    # Trace back the best path
    for cPosition in range(nPosition - 2, -1, -1):
        bestPath[cPosition] = parents[bestPath[cPosition + 1], cPosition + 1]

    return bestPath
