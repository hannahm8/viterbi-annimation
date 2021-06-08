"""
Viterbi annimation. 

Viterbi implementation in viterbi.viterbi_pathfinder based on 
J. Gardner's version for CW in the Lab project:
https://github.com/daccordeon/gravexplain
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

import viterbi


def plotPathsToNow(windowRefGrid,i,j):
  for jj in range(j):
    starti,startj = windowRefGrid[i,jj],jj-1
    nowi,nowj = i,jj
    plt.plot(([starti,nowi]),([startj+1,nowj+1]),color=linesColour)
  return None



def getImageName(number):
    """
    getting image names right
    """

    if   number<10:  imageName='00{}'.format(number)
    elif number<100: imageName='0{}'.format(number)
    else: imageName='{}'.format(number) 

    return imageName


def main():

    # where to save things 
    saveDir = 'tmp-images'
    os.mkdir(saveDir)
    #makedir 

    # set up colour choices and image type
    colourMap = cm.copper #cm.winter
    linesColour='#CDCDCD'
    pathColour = '#F18F01'#'#E2F1AF'#FBBA18'
    imageType='png'



    # making the data 
    width=30
    height = 40
    length = width*height
    data = [np.random.rand() for i in range(length)]
    data = np.atleast_1d(data)
    grid = data.reshape((width,height))


    # make the injection path 
    # hard coded, has a couple of steps bigger than one
    ipath = np.arange(0,height,1)
    fpath =    ([10,11,12,12,13,15,14,15,14,13,12,12,11,10,9,10,9,8,7,8,9,10,11,10,9,8,7,6,5,4,5,6,4,7,8,9,10,9,8,8])
    
    for ip,fp in zip(ipath,fpath):
        grid[fpath,ipath] = grid[fpath,ipath]+(grid[fpath,ipath]*0.03)



    # do viterbi 
    scoreGrid, pathGrid, windowRefGrid, pathEnd = viterbi.viterbi_pathfinder(grid)


    # get ready to make plot in scatter plot - probably a better way than this! 
    iis = []
    jjs = []
    grids=[]
    for i in range(len(grid[:,0])):
        for j in range(len(grid[0,:])):
            iis.append(i)
            jjs.append(j)
            grids.append(grid[i,j])



    #getPath 
    iPath = []
    jPath = []


    # iterate over time steps
    for i in range(len(pathGrid[0,:])):
        for j in range(len(pathGrid[:,0])):
           if pathGrid[j,i]==1.0:
              jPath.append(j) 
              iPath.append(i)
    iPath.append(i)
    jPath.append(pathEnd)
   

    print('Making frames in ./{}/'.format(saveDir))
    # save the first plot of the data only without tracks. 
    plt.clf()
    fig, ax = plt.subplots()
    y,x = np.meshgrid(np.linspace(-.5,height-.5,height+1),np.linspace(-.5,width-.5,width+1))
    z = grid
    c = ax.pcolormesh(x,y,z,cmap=colourMap)
    ax.set_aspect('equal')
    ax.set_xlim(-.5,width-.5)
    ax.set_ylim(-.5,height-.5)
    plt.axis('off')
    plt.savefig('{}/000.{}'.format(saveDir,imageType),dpi=150,bbox_inches='tight')

    counter = 1
    for j in range(len(grid[0,:])):
    
        """
        loop over to add all selected paths to plot
        """

        for i in range(len(grid[:,0])):
            """ 
            add steps one by one
            """
            if j!=(height-1):
                starti,startj = windowRefGrid[i,j],j-1
                nowi,nowj = i,j
                plt.plot(([starti,nowi]),([startj+1,nowj+1]),color=linesColour)

        imageName=getImageName(counter)
        plt.savefig('{}/{}.{}'.format(saveDir,imageName,imageType),dpi=150,bbox_inches='tight')
        counter+=1        
        

    # next add steps of the Viterbi path in reverse
    iReversedPath = list(reversed(iPath))
    jReversedPath = list(reversed(jPath))
    p = 0
    for p in range(len(iReversedPath)-1):
      
        jss = ([jReversedPath[p+1],jReversedPath[p]])
        iss = ([iReversedPath[p+1],iReversedPath[p]])
        plt.plot(jss,iss,color=pathColour,lw=3) 

        imageName=getImageName(counter)
        plt.savefig('{}/{}.{}'.format(saveDir,imageName,imageType),dpi=150,bbox_inches='tight')
        counter+=1



    plt.clf()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x,y,z,cmap=colourMap)
    plt.plot(jReversedPath,iReversedPath,color=pathColour,lw=3)
    ax.set_aspect('equal')
    ax.set_xlim(-.5,width-.5)
    ax.set_ylim(-.5,height-.5)
    plt.axis('off')

    # save five final images
    for i in range(5):
        """
        saves 5 identical image to make pause in gif 
        """
        imageName=getImageName(counter)
        plt.savefig('{}/{}.{}'.format(saveDir,imageName,imageType),dpi=150,bbox_inches='tight')
        counter+=1

    
    # make the gif  
    gifName = 'viterbiAnnimation.gif'
    print('Making gif at {}'.format(gifName))
    os.system("convert -delay 10 {}/*.png {}".format(saveDir,gifName))


if __name__ == "__main__":
    main()




