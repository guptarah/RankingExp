"""
Author: Rahul Gupta
This script trains separate Rankers on each person's preference list
and then sums up in unweighted fashion
"""

__docformat__ = 'restructedtext en'

import numpy
import scipy
import theano
import theano.tensor as T
import sys
sys.path.append('/auto/rcf-proj/pg/guptarah/RankingExp/scripts/DataPreparation')
import NoisyDataCreation.KendallTau as KendallTau

class SVRanker(object):
   """
   This function trains a SVRanker by minimizing the function stated in http://machinelearning.org/archive/icml2008/papers/448.pdf in equation (4)
   """

   def __init__(self,input,n_in,n_out):
      """ Initialize the parameters of the logistic regression
      
      :type input: theano.tensor.TensorType
      :param input: symbolic variable that describes input of the architecture

      :type n_in: int
      :param n_in: Feature dimensionality

      :type n_out: int
      :param n_out: Label dimensionality (1 in our case)

      """

      # initialize with 0 weights W
      self.W = theano.shared(value = .001*numpy.ones((n_in,n_out),
               dtype=theano.config.floatX), name='W', borrow=True)
      self.b = theano.shared(value = .001*numpy.ones((n_out,),
               dtype=theano.config.floatX), name='b', borrow=True)

      # parameters of the model
      self.params = [self.W, self.b]

      # keep track of model input
      self.input = input

      # symbolic expression for computing distance from class hyperplanes
      self.hyperplane_dist = T.dot(input,self.W) + self.b 

   def hinge_loss(self):
      """
      Gives the sum of distances from hyperplane of feature differences if 
      they are on the wrong side of the hyperplane
      """
      hinge_distances = T.ones_like(self.hyperplane_dist) - self.hyperplane_dist
      sum_hinge_distances = (T.sum(hinge_distances[hinge_distances > 0]) / T.shape(hinge_distances)[0]) + \
      .01*(T.dot(self.W.T,self.W) + T.dot(self.b,self.b))
      return sum_hinge_distances[0][0]


def gd_optimization(features,learning_rate=0.13, n_epochs=100):
   """
   We will perform gradient descent on the hinge loss in this part of the function

   :type features: float
   :param features: These are the features extracted by performing greater rank features - smaller rank features
            Please look at /auto/rcf-proj/pg/guptarah/RankingExp/scripts/DataPreparation/PairwiseFeatureCreation
            for how features are created 

   :type learning_rate: float
   :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

   :type n_epochs: int
   :param n_epochs: maximal number of epochs to run the optimizer
   """ 
   
   x = T.matrix('x')
   feature_dimensionality = features.shape[1]
   ranker = SVRanker(input=x, n_in=feature_dimensionality, n_out=1)
   cost = ranker.hinge_loss()

   g_W = T.grad(cost=cost, wrt=ranker.W)
   g_b = T.grad(cost=cost, wrt=ranker.b)

   updates = [(ranker.W, ranker.W - learning_rate * g_W),
              (ranker.b, ranker.b - learning_rate * g_b)]

   train_ranker = theano.function(inputs = [x], outputs = cost, updates = updates)

   epoch = 0
   while epoch < n_epochs:
      epoch = epoch + 1
      cost = train_ranker(features)
      #print cost

   return ranker.W.get_value(), ranker.b.get_value() 

if __name__ == "__main__":
   if len(sys.argv) != 2:
      print "Usage error: python TrainSVR.py FeatureFile"
   else:
      features_file = sys.argv[1]
      features = numpy.genfromtxt(features_file,delimiter=',',dtype='float')
      gd_optimization(features) 
