"""
This code instead of minimizing [1-z{w.T*(x_g-x_l)}] minimizes k*[1-z{w.T*(x_g-x_l)}] + (1-k)[1-z{w.T*(x_l-x_g)}]
where k is the probability that x_g is higher ranked than x_l
"""

import numpy
import sys

def get_gradient(features,w,soft_scores):
   D = features.shape[1]
   N = features.shape[0]
   hyperplane_dist = numpy.dot(features,w.T)
   hinge_dist = numpy.array(numpy.ones((hyperplane_dist.shape)) - hyperplane_dist) 
   wtd_features = numpy.multiply(features,numpy.tile(soft_scores,(1,D)))
   grad = -1*numpy.sum(wtd_features[hinge_dist[:,0]>0,:],axis=0)/N
   return grad

def svr_optimization(features,soft_scores,w_init=0,learning_rate=0.1,n_epochs=100,lambda_w=.01):

   D = features.shape[1]
   N = features.shape[0]
   if w_init.shape[1] != D+1:
      print 'Feature dimensionality and W dimensionality mismatch'

   if soft_scores.shape[0] != N:
      print 'count of soft scores is not equal to number of datapoints'

   ext_features_g = numpy.hstack((features,numpy.ones((features.shape[0],1))))
   ext_features_l = numpy.hstack((-1*features,numpy.ones((features.shape[0],1)))) 

   w = w_init 
   for i in range(n_epochs): 
#      # computing cost for k*[1-z{w.T*(x_g-x_l)}] part
#      hyperplane_dist_g = numpy.dot(ext_features_g,w.T)
#      wtd_hinge_dist_g = numpy.array(numpy.multiply(soft_scores, (numpy.ones((hyperplane_dist_g.shape)) - hyperplane_dist_g)))
#      cost_g = numpy.sum(wtd_hinge_dist_g[wtd_hinge_dist_g[:,0]>0])/N + lambda_w*numpy.dot(w,w.T)
#      wtd_ext_features_g = numpy.multiply(ext_features_g,numpy.tile(soft_scores,(1,D+1)))
#      grad_g = -1*numpy.sum(wtd_ext_features_g[wtd_hinge_dist_g[:,0]>0,:],axis=0)/N 
#
#      # computing cost for (1-k)[1-z{w.T*(x_l-x_g)}] part
#      hyperplane_dist_l = numpy.dot(ext_features_l,w.T)
#      soft_scores_l = numpy.ones((soft_scores.shape))-soft_scores
#      wtd_hinge_dist_l = numpy.array(numpy.multiply(soft_scores_l, (numpy.ones((hyperplane_dist_l.shape)) - hyperplane_dist_l)))
#      cost_l = numpy.sum(wtd_hinge_dist_l[wtd_hinge_dist_l[:,0]>0])/N + lambda_w*numpy.dot(w,w.T)
#      wtd_ext_features_l = numpy.multiply(ext_features_l,numpy.tile(soft_scores_l,(1,D+1)))
#      grad_l = -1*numpy.sum(wtd_ext_features_l[wtd_hinge_dist_l[:,0]>0,:],axis=0)/N 
#
#      grad = grad_l + grad_g + 2*lambda_w*w
      
      grad_g = get_gradient(ext_features_g,w,soft_scores)
      soft_scores_l = numpy.ones(soft_scores.shape) - soft_scores
      grad_l = get_gradient(ext_features_l,w,soft_scores_l)
      grad = grad_l + grad_g + 2*lambda_w*w
      w = w - learning_rate*grad

   return w 

if __name__ == "__main__":
   if len(sys.argv) != 4:
      print "Usage error: python TrainSVR.py FeatureFile SoftScoreFile WInitFile"
   else:
      features_file = sys.argv[1]
      softscore_file = sys.argv[2] # this score contains confidence if diff as computed is 
      # write or wrong, i.e., k in header of the file
      w_init_file = sys.argv[3] # this score contains confidence if diff as computed is 
      features = numpy.genfromtxt(features_file,delimiter=',',dtype='float')
      soft_scores = numpy.genfromtxt(softscore_file,delimiter=',',dtype='float')
      w_init = numpy.matrix(numpy.genfromtxt(w_init_file,delimiter=',',dtype='float'))
      print w_init.shape
      svr_optimization(features,soft_scores,w_init) 
