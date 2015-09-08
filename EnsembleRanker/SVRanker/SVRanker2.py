import numpy
import theano
import theano.tensor as T
import sys
rng = numpy.random


def svr_optimization(features,learning_rate=0.1,n_epochs=100):

   D = features.shape[1]
   N = features.shape[0]

   x = T.dmatrix("x")
   w = theano.shared(rng.randn(D), name="w")
   b = theano.shared(0., name="b")
   print "Initial model"
   print w.get_value(), b.get_value()

   # Theano expression graph
   hyperplane_dist = T.dot(x,w) - b
   hinge_distances = T.ones_like(hyperplane_dist) - hyperplane_dist
   #cost = (T.sum(hinge_distances[hinge_distances > 0]) / T.shape(hinge_distances)[0]) + \
   #.01*(T.dot(w.T,w) + T.dot(b,b))
   cost = T.sum(T.exp(.0000001* hinge_distances))  
   #.01*(T.dot(w.T,w) + T.dot(b,b))
   gw, gb = T.grad(cost, [w, b])

   # Compile
   train = theano.function(inputs = [x], outputs = cost, 
            updates=((w, w - learning_rate * gw), (b, b - learning_rate * gb)))

   # Train
   for i in range(n_epochs):
      obt_cost = train(features)
      print obt_cost 

   return w.get_value(),b.get_value()

if __name__ == "__main__":
   if len(sys.argv) != 2:
      print "Usage error: python TrainSVR.py FeatureFile"
   else:
      features_file = sys.argv[1]
      features = numpy.genfromtxt(features_file,delimiter=',',dtype='float')
      svr_optimization(features) 
