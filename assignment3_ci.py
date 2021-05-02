

import math
import numpy as np 
import copy
import random
from PIL import Image, ImageDraw

def ValueIteration(grid, gamma, theta):
  x = len(grid)
  y = len(grid[0])
  
  reward = copy.deepcopy(grid)
  #initializing values with the default reward values
  value = copy.deepcopy(grid)
  #initializing the policy grid
  policy = copy.deepcopy(grid)

  #list of all the states available for easy access during iteration
  states = []
  for i in range(x):
    for j in range(y):
      states.append((i,j))

  iter_ = 0
  #mapping the directions to an integer
  directions ={
      3: 'D',
      2: 'U',
      1: 'L',
      0: 'R'
  }

  #value iteration
  while True:
    iter_ +=1
    delta = 0
    for i, j in states:
      if reward[i][j] == -1:
        u = value[i][j]
        tmpList = []
        if i != x-1:
          tmpList.append((reward[i][j] + gamma*value[i+1][j] , 3))
        if i != 0:
          tmpList.append((reward[i][j] + gamma*value[i-1][j], 2))        
        if j!= 0:
          tmpList.append((reward[i][j] + gamma*value[i][j-1], 1))
        if j != y-1:
          tmpList.append((reward[i][j] + gamma*value[i][j+1], 0))
        value[i][j] = max(tmpList)[0]
        policy[i][j] = directions[max(tmpList)[1]]
        delta = max([delta , np.abs(u - value[i][j])])

    #check to see if the policy is converged
    if delta < theta:
      
      break
  return policy,iter_



def visualize(policy):
    size = (len(policy), len(policy[0]))
    height = 500
    width = 500
    steps = size[1]
    stepSize = int(width/steps)
    arrow = Image.open("arrow.png").convert("RGBA")
    arrow = arrow.resize((stepSize, stepSize))
    image = Image.new(mode='RGBA', size=(width, height), color=255)
    draw = ImageDraw.Draw(image)

    for x in range(size[0]):
        for y in range(size[1]):
            if policy[x][y] == -100:
                draw.rectangle([(y*stepSize, x*stepSize), ((y+1)*stepSize, (x+1)*stepSize)], fill="#FF0000")
            elif policy[x][y] == 100:
                draw.rectangle([(y*stepSize, x*stepSize), ((y+1)*stepSize, (x+1)*stepSize)], fill="#00FF00")
            else:
                if policy[x][y] == "R":
                    rotation = 0
                if policy[x][y] == "U":
                    rotation = 90
                if policy[x][y] == "L":
                    rotation = 180
                if policy[x][y] == "D":
                    rotation = 270
                arrowDirection = arrow.rotate(rotation)
                image.paste(arrowDirection, (y*stepSize, x*stepSize), arrowDirection)

    image.show()
    return image


def generateGrid(size=(10, 10)):
    grid = [[0 for i in range(size[1])] for i in range(size[0])]
    reward_x = random.randint(0,size[0]-1)
    reward_y = random.randint(0,size[1]-1)
    for i in range(size[0]):
        for j in range(size[1]):
            r = random.random()
            if r <= 0.1:
                grid[i][j] = -100
            else:
                grid[i][j] = -1
    
    grid[reward_x][reward_y] = 100

    return grid


grid = []
grid.append([-1 for _ in range(10)])
grid.append([-1 for _ in range(10)])
grid.append([-1,-1,-1,-1,-100,-100,-100,-100,-1,-100])
grid.append([-1 for _ in range(10)])
grid.append([-1 for _ in range(10)])
grid.append([-1 for _ in range(10)])
grid.append([-1,-1,-100, -1,-1,-1,-1,-1,-1,-1])
grid.append([-1,-1,-100,-100,-1,-1,-100,-100,-100,-1])
grid.append([-1,-1,-100, -1,-1,-1,-1,-1,-1,-1])
grid.append([-1,-1,-100,-100,-100,-100,-100,-100,-100,100])


# for random input uncomment the line below
# grid = generateGrid((10,10))


grid1, iteration = ValueIteration(grid, 0.9, 0.1)
print('Number of iterations are',iteration)


visualize(grid1)