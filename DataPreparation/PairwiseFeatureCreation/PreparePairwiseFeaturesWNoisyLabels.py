"""
Author: Rahul Gupta
Given a feature set and corresponding ranks, this function creates a dataset of x1-x2 where the rank of x1 > x2.

Also given annotator preferences, this also creates a vector of 0 and 1, where 0 indicates disagreement with
the true labels and otherwise
"""

import numpy
import sys
import os 


def CreateFiles(labels_file,noisy_labels_dir,noisy_labels_cnt,features_file,qid_file):
   labels = numpy.genfromtxt(labels_file,dtype='float')
   noisy_labels_all = numpy.empty((noisy_labels_cnt,0)).tolist()
   for i in range(noisy_labels_cnt):
      noisy_labels_file = noisy_labels_dir + '/noisy_labels' + str(i+1)
      cur_noisy_labels = numpy.genfromtxt(noisy_labels_file,dtype='float')
      noisy_labels_all[i] = cur_noisy_labels
   features = numpy.genfromtxt(features_file,dtype='float',delimiter=',')
   qid = numpy.genfromtxt(qid_file,dtype='int')
   qid_unique = numpy.unique(qid)

   labels_dir = os.path.dirname(os.path.abspath(labels_file))

   for qid_cur in qid_unique:
      print 'current qid: ',qid_cur
      save_dir = labels_dir + '/noisy_features/' + os.path.basename(labels_file)
      noisy_labels_save_dir = labels_dir + '/noisy_labels_pairwise/' 
      if not os.path.exists(save_dir):
         os.makedirs(save_dir)
      if not os.path.exists(noisy_labels_save_dir):
         os.makedirs(noisy_labels_save_dir)

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

      for i in range(noisy_labels_cnt):
         save_file = noisy_labels_save_dir + '/' + str(qid_cur) + '.noisy_labels' + str(i+1)
         print save_file
         f_save = file(save_file,'a')
         noisy_labels_cur = noisy_labels_all[i] [qid == qid_cur]
         for j in unique_labels:
            noisy_labels_cur_j = noisy_labels_cur[labels_cur == j]
            noisy_labels_cur_greater_j = noisy_labels_cur[labels_cur > j]

            for k in range(0,noisy_labels_cur_j.shape[0]):
               for l in range(0,noisy_labels_cur_greater_j.shape[0]):
                  if noisy_labels_cur_j[k] > noisy_labels_cur_greater_j[l]:
                     numpy.savetxt(f_save,numpy.zeros((1,1)),fmt='%f')
                  else:
                     numpy.savetxt(f_save,numpy.ones((1,1)),fmt='%f')
         f_save.close() 

if __name__ == "__main__":
   if len(sys.argv) != 6:
      print "Wrong number of arguments\
      Usage: python PreparePairwiseFeatures LabelsFile NoisyLabelsDir noisy_labels_cnt FeaturesFile QidFile"
   else:
      labels_file = sys.argv[1]
      noisy_labels_dir = sys.argv[2] # where noisy labels are stored
      noisy_labels_cnt = int(sys.argv[3]) # count of annotators
      features_file = sys.argv[4]
      qid_file = sys.argv[5]
      CreateFiles(labels_file,noisy_labels_dir,noisy_labels_cnt,features_file,qid_file) 
     
