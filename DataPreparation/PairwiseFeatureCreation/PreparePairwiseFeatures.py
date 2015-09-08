"""
Author: Rahul Gupta
Given a feature set and corresponding ranks, this function creates a dataset of x1-x2 where the rank of x1 > x2.
We will later optimize the hinge loss function as described in http://machinelearning.org/archive/icml2008/papers/448.pdf in equation (4)
Given features from this function, the z in equation (4) will always be 1
"""

import numpy
import sys
import os 


def CreateFiles(labels_file,features_file,qid_file):
   labels = numpy.genfromtxt(labels_file,dtype='float')
   features = numpy.genfromtxt(features_file,dtype='float',delimiter=',')
   qid = numpy.genfromtxt(qid_file,dtype='int')
   qid_unique = numpy.unique(qid)

   labels_dir = os.path.dirname(os.path.abspath(labels_file))

   for qid_cur in qid_unique:
      print 'current qid: ',qid_cur
      save_dir = labels_dir + '/noisy_features/' + os.path.basename(labels_file)
      if not os.path.exists(save_dir):
         os.makedirs(save_dir)

      save_file = save_dir + '/' + str(qid_cur) + '.features'
      f_save = file(save_file,'a')

      labels_cur = labels[qid == qid_cur]
      features_cur = features[qid == qid_cur,:]
      unique_labels = numpy.unique(labels)

      qid_compare_features = []
      for i in unique_labels:
         features_i = features_cur[labels_cur == i,:]
         features_greater_i = features_cur[labels_cur > i,:]

         for j in range(0,features_i.shape[0]):
            for k in range(0,features_greater_i.shape[0]):
               feature_difference = numpy.matrix(features_greater_i[k,:] - features_i[j,:])
               numpy.savetxt(f_save,feature_difference,delimiter=',',fmt='%f')

      f_save.close()

if __name__ == "__main__":
   if len(sys.argv) != 4:
      print "Wrong number of arguments\
      Usage: python PreparePairwiseFeatures LablesFile FeaturesFile QidFile"
   else:
      labels_file = sys.argv[1]
      features_file = sys.argv[2]
      qid_file = sys.argv[3]
      CreateFiles(labels_file,features_file,qid_file) 
     
