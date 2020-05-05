import numpy as np
import cv2
from __main__ import *
import matplotlib.pyplot as plt
import imutils
import os
import random


from som import *
from som2 import *
from som3 import *

from prb1 import*
from prb2 import*
from prb3 import*


print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = False
prgRun = True



def main(prgRun):

    problem=2
    problem = int(input('Which problem would you like to run?\nEnter 1 for TSP\nEnter 2 for Wine\nEnter 3 for Polygon\nPlease enter your choice :'))


    if problem==1:
        prb1main()
    if problem==2:
        prb2main()
    if problem==3:
        prb3main()


    prgRun = False
    return prgRun


print('Function Initializations complete')

if __name__ == '__main__':
    print('Start')
    prgRun = True
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()
