"""
Function to train using EM

E step: estimating that given feat diff is correctly calculated (called k in below)
k*[1-z{w.T*(x_g-x_l)}] + (1-k)[1-z{w.T*(x_l-x_g)}]

M step: estimating the model parameters w and A matrices
"""

def InitializeK(N):
   init_k = numpy.ones((N,2))
   init_k[:,1] = 0
   return init_k

def InitializeW(D):
   return numpy.ones((D,1)) 

def InitializeA():
   A[0] = 0 # probability that he flipped
   A[0] = 1 # probability he did not flip
   return A

def SigmoidProb(ext_diff_feats,w):
   term1 = numpy.exp(numpy.dot(ext_diff_feats,w.T))
   term2 = term1 + ones(term1.shape)
   return numpy.divide(term1,term2) 

def ComputeA(k,cur_annt_labels):
   e1_probs = k
   e2_probs = numpy.ones(k.shape) - k 
   cur_not_flip_prob = numpy.sum(e1_probs[cur_annt_labels == 1])/numpy.sum(e1_probs) +\
      numpy.sum(e0_probs[cur_annt_labels == 0])/numpy.sum(e0_probs) 
   cur_flip_prob = numpy.sum(e1_probs[cur_annt_labels == 0])/numpy.sum(e1_probs) +\
      numpy.sum(e0_probs[cur_annt_labels == 1])/numpy.sum(e0_probs)
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
      
      # E step. Estimating k
      model_probs = SigmoidProb(ext_diff_feats, init_w)
      prod_probs_E1 = model_probs # probability that assumed diff is correct 
      prod_probs_E0 = ones(model_probs.shape) - model_probs # probability that assumed diff is incorrect
      for i in range(R):
         A_cur = A[i]
         A_cur_replicated = numpy.matlib.repmat(A_cur,N,1)
         cur_annt_labels = annt_comparison_labels[i]       
 
         # Below for E1 if an annotator said 0, flipping probability is multiplied and otherwise
         cur_annt_probs_E1 = numpy.choose(cur_annt_labels,A_cur_replicated.T) 
         prod_probs_E1 = prod_probs * cur_annt_probs

         # Below for E0 if an annotator said 1, flipping probability is multiplied and otherwise
         cur_annt_probs_E0 = numpy.choose(numpy.logical_not(cur_annt_labels),A_cur_replicated.T)
         prod_probs_E0 = prod_probs * cur_annt_probs

      k = numpy.divide(prod_probs_E1,(prod_probs_E1+prod_probs_E0))


      # M step. 
      # Estimating w 
      diff_feats = ext_diff_feats[:,:-2] # unfortunately ones are appended again in SVRankerSoft 
      learning_rate = 0.2
      n_epochs = 2000
      lambda_w = .01  
      w = SVRankerSoft.svr_optimization(ext_diff_feats,k,w,learning_rate,n_epochs,lambda_w)

      # Estimating A's
      for i in range(R):
         cur_annt_labels = annt_comparison_labels[i]     
         A[i] = ComputeA(k,cur_annt_labels) 
      
      if iter_counter > max_iter:
         convergence_flag = 0 
