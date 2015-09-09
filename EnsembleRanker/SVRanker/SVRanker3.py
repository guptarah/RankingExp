import numpy
import sys


def svr_optimization(features,w_init,learning_rate=0.1,n_epochs=100,lambda_w=.01):

   D = features.shape[1]
   N = features.shape[0]
   if w_init.shape[1] != D+1:
      print 'Feature dimensionality and W dimensionality mismatch'
   ext_features = numpy.hstack((features,numpy.ones((features.shape[0],1))))
   
   w = w_init 
   for i in range(n_epochs): 
      hyperplane_dist = numpy.dot(ext_features,w.T)
      hinge_dist = numpy.ones((hyperplane_dist.shape)) - hyperplane_dist
      cost = numpy.sum(hinge_dist[hinge_dist[:,0]>0])/N + lambda_w*numpy.dot(w,w.T)
      grad = -1*numpy.sum(ext_features[hinge_dist[:,0]>0,:],axis=0)/N + 2*lambda_w*w
      w = w - learning_rate *grad
      #print cost
 
   return w 

if __name__ == "__main__":
   if len(sys.argv) != 2:
      print "Usage error: python TrainSVR.py FeatureFile"
   else:
      features_file = sys.argv[1]
      features = numpy.genfromtxt(features_file,delimiter=',',dtype='float')
      svr_optimization(features) 
