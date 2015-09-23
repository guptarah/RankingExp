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

def InitializeK(N):
   init_k = .5*numpy.ones((N,1))
   return init_k

def InitializeW(D):
   return numpy.ones((1,D)) 

def InitializeA():
   A = [0.1,0.9]# [probability that he flipped
   # probability he did not flip]
   return A

def SigmoidProb(ext_diff_feats,w):
   term1 = numpy.exp(numpy.dot(ext_diff_feats,w.T)*(.1/(numpy.dot(w,w.T)+.1)))
   term1[numpy.isinf(term1)] = 1000
   term2 = term1 + numpy.ones(term1.shape)
   return numpy.divide(term1,term2) 

def ComputeA(k,cur_annt_labels):
   e1_probs = k
   e0_probs = numpy.ones(k.shape) - k 
#   cur_not_flip_prob = numpy.sum(e1_probs[cur_annt_labels == 1])/numpy.sum(e1_probs) +\
#      numpy.sum(e0_probs[cur_annt_labels == 0])/numpy.sum(e0_probs) 
#   cur_flip_prob = numpy.sum(e1_probs[cur_annt_labels == 0])/numpy.sum(e1_probs) +\
#      numpy.sum(e0_probs[cur_annt_labels == 1])/numpy.sum(e0_probs)
#   sum_probs = cur_flip_prob + cur_not_flip_prob
#   cur_not_flip_prob = numpy.sum(e1_probs[cur_annt_labels == 1]) + numpy.sum(e0_probs[cur_annt_labels == 0])
#   cur_flip_prob = numpy.sum(e1_probs[cur_annt_labels == 0]) + numpy.sum(e0_probs[cur_annt_labels == 1])
#   sum_probs = cur_flip_prob + cur_not_flip_prob
#   A = [cur_flip_prob/sum_probs, cur_not_flip_prob/sum_probs]
   k_disc = k > 0.5
   agreements = numpy.equal(numpy.matrix(cur_annt_labels).T,k_disc)
   cur_not_flip_prob = numpy.mean(agreements)
   cur_flip_prob = 1 - cur_not_flip_prob 
   A = [cur_flip_prob, cur_not_flip_prob]
   return A

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
      A[i] = InitializeA()


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
         cur_annt_probs_E1 = numpy.matrix(A_cur) * cur_label_mat 
         prod_probs_E1 = numpy.multiply(prod_probs_E1,cur_annt_probs_E1.T)

         # Below for E0 if an annotator said 1, flipping probability is multiplied and otherwise
         cur_annt_probs_E0 = numpy.matrix(A_cur) * numpy.logical_not(cur_label_mat) 
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
         A[i] = ComputeA(k,cur_annt_labels)
      if iter_counter > max_iter:
         convergence_flag = 0

   print 'Finished training'      
   for i in range(R):
      print A[i] 

   return w,k

