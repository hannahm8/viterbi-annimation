import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def viterbi_pathfinder(grid, scanning_range=1):
    """
    Using James Gardner's viterbi implementation from https://github.com/daccordeon/gravexplain
    """

    """
    find the highest scoring path through the grid, left-to-right,
    as by the viterbi algorithm, with connections plus-minus the scanning_range;
    returns score grid for best path to each node and a bitmap of the total best path
    """
    # normalised grid, algorithm goal is to maximise product of values
    ngrid  = grid/np.max(grid)
    # logarithm avoids underflow, equvivalent to maximise sum of log of values
    lngrid = np.log(ngrid)
    
    long_timesteps = grid.shape[1]

    # keep track of running scores for best path to each node
    score_grid  = np.copy(lngrid)
    pathfinder_flag = len(lngrid[:,0])
    # pathfinder stores the survivor paths, i.e. the previous best step
    # to allow back-tracking to recover the best total path at the end
    pathfinder = np.full(np.shape(lngrid), pathfinder_flag)
    # pathfinder flag+1 for reaching the first, 0-index column        
    pathfinder[:,0] = pathfinder_flag+1       


    window_ref_grid = np.zeros((len(grid[:,0]),long_timesteps-1))


    # implementation of the viterbi algorithm itself
    # finding the best path to each node, through time
    # see: https://www.youtube.com/watch?v=6JVqutwtzmo
    for j in range(1,long_timesteps):
        for i in range(len(score_grid[:,j])):
            # index values for where to look relative to i in previous column
            k_a = max(0, i-scanning_range) 
            k_b = min(len(score_grid[:,j-1])-1,
                      i+scanning_range)
            window = score_grid[:,j-1][k_a:k_b+1]
            # find the best thing nearby in the previous column ...
            window_score = np.max(window)
            window_ref   = k_a+np.argmax(window)
            # ... and take note of it, summing the log of values
            score_grid[i][j] += window_score
            pathfinder[i][j] = window_ref 
            #print('window socre ',i, window_ref)
            window_ref_grid[i][j-1] = window_ref
            

    # look at the very last column, and find the best total ending ...
    best_score  = np.max(score_grid[:,-1])
    best_end = np.argmax(score_grid[:,-1])
    # ... and retrace its steps through the grid
    best_path_back = np.full(long_timesteps,pathfinder_flag+2)
    best_path_back[-1] = best_end
    # best_path_back is the viterbi path, the highest scoring overall 
    # path_grid is the binary image of the viterbi path taken
    path_grid = np.zeros(np.shape(ngrid))
    tmp_path = pathfinder[best_end][-1]

    for j in reversed(range(0,long_timesteps-1)):
        path_grid[tmp_path][j] = 1
        # take pathfinder value in current step and follow it backwards
        best_path_back[j] = tmp_path    
        tmp_path = pathfinder[tmp_path][j]

    # make sure we got all the way home
    # (that the retrace found the initial edge)
    assert tmp_path == pathfinder_flag+1
    
    return score_grid, path_grid, window_ref_grid, best_end



