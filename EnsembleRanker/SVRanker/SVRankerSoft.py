"""
This code instead of minimizing [1-z{w.T*(x_g-x_l)}] minimizes k*[1-z{w.T*(x_g-x_l)}] + (1-k)[1-z{w.T*(x_l-x_g)}]
where k is the probability that x_g is higher ranked than x_l
"""

import numpy
import sys


def svr_optimization(features,soft_scores,w_init=0,learning_rate=0.1,n_epochs=100,lambda_w=.01):

   D = features.shape[1]
   N = features.shape[0]
   if w_init.shape[1] != D+1:
      print 'Feature dimensionality and W dimensionality mismatch'

   if soft_scores.shape[0] != N:
      print 'count of soft scores is not equal to number of datapoints'

   ext_features = numpy.hstack((features,numpy.ones((features.shape[0],1))))
   
   w = w_init 
   for i in range(n_epochs): 
      hyperplane_dist = numpy.dot(ext_features,w.T)
      hinge_dist = numpy.ones((hyperplane_dist.shape)) - hyperplane_dist

      # computing cost for k*[1-z{w.T*(x_g-x_l)}] part
      hyperplane_dist_g = numpy.dot(ext_features,w.T)
      wtd_hinge_dist_g = numpy.multiply(soft_scores, (numpy.ones((hyperplane_dist_g.shape)) - hyperplane_dist_g))
      cost_g = numpy.sum(wtd_hinge_dist_g[wtd_hinge_dist_g[:,0]>0])/N + lambda_w*numpy.dot(w,w.T)
      grad_g = -1*numpy.sum(ext_features[wtd_hinge_dist_g[:,0]>0,:],axis=0)/N + 2*lambda_w*w 

      # computing cost for (1-k)[1-z{w.T*(x_l-x_g)}] part
      hyperplane_dist_l = -1*numpy.dot(ext_features,w.T)
      soft_scores_l = numpy.ones((soft_scores.shape))-soft_scores
      wtd_hinge_dist_l = numpy.multiply(soft_scores_l, (numpy.ones((hyperplane_dist_l.shape)) - hyperplane_dist_l))
      cost_l = numpy.sum(wtd_hinge_dist_l[wtd_hinge_dist_l[:,0]>0])/N + lambda_w*numpy.dot(w,w.T)
      grad_l = -1*numpy.sum(ext_features[wtd_hinge_dist_l[:,0]>0,:],axis=0)/N + 2*lambda_w*w 

      w = w - learning_rate*grad_g - learning_rate*grad_l
      #print cost
 
   return w 

if __name__ == "__main__":
   if len(sys.argv) != 3:
      print "Usage error: python TrainSVR.py FeatureFile SoftScoreFile"
   else:
      features_file = sys.argv[1]
      softscore_file = sys.argv[2] # this score contains confidence if diff as computed is 
      # write or wrong, i.e., k in header of the file
      features = numpy.genfromtxt(features_file,delimiter=',',dtype='float')
      soft_scores = numpy.genfromtxt(softscore_file,delimiter=',',dtype='float')
      svr_optimization(features,soft_scores) 
