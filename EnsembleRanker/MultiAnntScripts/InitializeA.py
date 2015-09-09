""" 
script to initialize the A matrix
"""

import numpy

def InitializeA():
   A = numpy.empty([1,2])
   A[0,0] = 1 # probability that he dint flip
   A[0,1] = 0 # probability he flipped
   return A
