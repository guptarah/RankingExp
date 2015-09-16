"""
Function to train using EM

E step: estimating that given feat diff is correctly calculated (called k in below)
k*[1-z{w.T*(x_g-x_l)}] + (1-k)[1-z{w.T*(x_l-x_g)}]

M step: estimating the model parameters w and A matrices
"""

import numpy
import sys
sys.path.append('/auto/rcf-proj/pg/guptarah/RankingExp/scripts/EnsembleRanker/SVRanker')
import SVRankerSoft
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn import preprocessing

def InitializeK(N):
   init_k = .5*numpy.ones((N,1))
   return init_k

def InitializeW(D):
   return numpy.ones((1,D)) 

def InitializeA(N):
   A = numpy.ones((2,N))
   A[0,:] = 0.4 # probability he flipped
   A[1,:] = 0.6 # probability he did not flip
   return A

def SigmoidProb(ext_diff_feats,w):
   term1 = numpy.exp((1/(numpy.dot(w,w.T)+.1))*.1*numpy.dot(ext_diff_feats,w.T))
   term1[numpy.isinf(term1)] = 1000
   term2 = term1 + numpy.ones(term1.shape)
   return numpy.divide(term1,term2) 

def ComputeAKNN(k,cur_annt_labels,features): # compute A using k means clustering
   # clustering based on Kmeans
   nclust = 2 
   init_centroids = numpy.zeros((nclust,features.shape[1]))
   k_disc = k > 0.5 # models decisions
   agreements = numpy.ravel(1*numpy.equal(numpy.matrix(cur_annt_labels).T,k_disc))
   features0 = features[agreements == 0,:]
   features1 = features[agreements == 1,:]
   init_centroids[0,:] = numpy.mean(features0,axis=0)
   for i in range (1,nclust):
      init_centroids[i,:] = numpy.mean(features1,axis=0)
   kmean_est = KMeans(n_clusters = nclust,init=init_centroids)
   kmean_est.fit(features)
   labels = kmean_est.labels_

   # getting the A matrix per cluster
   A_entries = numpy.zeros((2,nclust))
   for i in range(0,nclust):
      cur_agreements = agreements[labels == i]
      A_entries[1,i] = numpy.mean(cur_agreements) 
      A_entries[0,i] = 1 - A_entries[1,i] 

   # getting 1 in K encoding from labels
   lb = preprocessing.LabelBinarizer()
   lb.fit(labels)
   labels_encoding = lb.transform(labels).T
   if nclust == 2: # since then it only gives a vector of 0 and 1
      labels_encoding = numpy.vstack((numpy.logical_not(labels_encoding),labels_encoding))
   A = numpy.dot(A_entries,labels_encoding)
   return A

def ComputeA(k,cur_annt_labels,features): # compute A using logistic regression
   k_disc = k > 0.5 # models decisions
   agreements = numpy.ravel(1*numpy.equal(numpy.matrix(cur_annt_labels).T,k_disc))
   # training model to predict agreements based on features
   logreg = linear_model.LogisticRegression(penalty='l2',C=1)
   logreg.fit(features,agreements)
   A = logreg.predict_proba(features)
   return A.T

def TrainModel(ext_diff_feats,annt_comparison_labels,max_iter=100):
   """
   diff_feats: difference between features extended with ones
   annt_comparison_labels: comparison labels (0/1) showing if annotator said if x_g > x_l (labels it: 1) or x_g < x_l (labels it: 0)
   """  
   
   N = ext_diff_feats.shape[0] # number of data point comparisons  
   R = len(annt_comparison_labels) # number of annotators
   D = ext_diff_feats.shape[1] # feature dimensionality
 
   # Initialization
   k = InitializeK(N) 
   w = InitializeW(D)
   A = numpy.empty((R,0)).tolist()
   for i in range(R):
      A[i] = InitializeA(N)


   convergence_flag = 1
   iter_counter = 0
   while convergence_flag:
      iter_counter = iter_counter + 1 
      
      # E step. Estimating k
      model_probs = SigmoidProb(ext_diff_feats, w)
      prod_probs_E1 = model_probs # probability that assumed diff is correct 
      prod_probs_E0 = numpy.ones(model_probs.shape) - model_probs # probability that assumed diff is incorrect
      for i in range(R):
         A_cur = A[i]
         cur_annt_labels = annt_comparison_labels[i]       
         cur_label_mat = numpy.vstack((numpy.logical_not(cur_annt_labels),cur_annt_labels))         
 
         # Below for E1 if an annotator said 0, flipping probability is multiplied and otherwise
         cur_annt_probs_E1 = numpy.matrix(numpy.sum(numpy.multiply(A_cur,cur_label_mat),axis=0))
         prod_probs_E1 = numpy.multiply(prod_probs_E1,cur_annt_probs_E1.T)

         # Below for E0 if an annotator said 1, flipping probability is multiplied and otherwise
         cur_annt_probs_E0 = numpy.matrix(numpy.sum(numpy.multiply(A_cur,numpy.logical_not(cur_label_mat)),axis=0))
         prod_probs_E0 = numpy.multiply(prod_probs_E0,cur_annt_probs_E0.T)

      k_term1 = prod_probs_E1
      k_term2 = prod_probs_E1+prod_probs_E0 #+ .001*numpy.ones(prod_probs_E1.shape)
      k = numpy.divide(k_term1,k_term2)


      # M step. 
      # Estimating w 
      diff_feats = ext_diff_feats[:,:-1] # unfortunately ones are appended again in SVRankerSoft 
      learning_rate = 0.02
      n_epochs = 20
      lambda_w = .001  
      w = SVRankerSoft.svr_optimization(diff_feats,k,w,learning_rate,n_epochs,lambda_w)

      # Estimating A's
      for i in range(R):
         cur_annt_labels = annt_comparison_labels[i]     
         A[i] = ComputeAKNN(k,cur_annt_labels,ext_diff_feats) 
      if iter_counter > max_iter:
         convergence_flag = 0

   print 'Finished training'      
   for i in range(R):
      print numpy.mean(A[i], axis=1)

   return w,k

