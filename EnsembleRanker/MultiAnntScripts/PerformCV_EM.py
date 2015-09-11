"""
This script performs cv on EM algorithm that I have proposed
"""

import numpy
import scipy
import TrainEM

def PerformCV(qid_file, diff_feat_dir, feat_file, true_labels_file, noisy_labels_dir, batch_size=1, count_annts):
"""
diff_feat_dir: directory where the diff features are stored
noisy_labels_dir: directory where the noisy labels are stored
true_labels_file: the true labels file used for test set evaluation
qid_file: the qid file location
noisy_labels_dir: directory containing noisy preferences corresponding to
features in the diff_feat_dir
count_annts: number of annotators

Example values:
qid_file= '/auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/qids'
diff_feat_dir='/auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/noisy_features/labels/'
feat_file='/auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/features'
true_labels_file='/auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/labels'
noisy_labels_dir='/auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/noisy_labels_pairwise'
batch_size=1
count_annts=6

"""

   qids = numpy.genfromtxt(qid_file,dtype='int')
   qids_unique = numpy.unique(qids)

   features = numpy.genfromtxt(feat_file,delimiter=',')
   labels = numpy.genfromtxt(true_labels_file,delimiter=',')

   if numpy.remainder(len(qids_unique),batch_size):
      print "Please provide a split that divides number of unique qids"
      return

   num_batches = len(qids_unique)/batch_size

   for i in range(num_batches):
      # Determine the qids in test, dev and train sets
      test_id = i 
      test_batch_qids = qids_unique[numpy.arange(batch_size*test_id,batch_size*(test_id+1))]

      dev_id = numpy.remainder(i+1,num_batches)
      dev_batch_qids = qids_unique[numpy.arange(batch_size*dev_id,batch_size*(dev_id+1))]
      
      train_batch_qids = numpy.setdiff1d(qids_unique,numpy.union1d(test_batch_qids,dev_batch_qids))  
   
      # find the features and labels for the train and the dev set
      test_features = features[numpy.in1d(qids,test_batch_qids).T,:] 
      test_labels = numpy.matrix(labels[numpy.in1d(qids,test_batch_qids)]).T
      dev_features = features[numpy.in1d(qids,dev_batch_qids),:] 
      dev_labels = numpy.matrix(labels[numpy.in1d(qids,dev_batch_qids)]).T

      # get all train set features together
      train_diff_features = numpy.empty([0,test_features.shape[1]])
      annt_labels = numpy.empty([count_annts,0]).tolist()
      for train_batch_qid in train_batch_qids:
         feature_diff_file = diff_feat_dir + '/' + str(int(train_batch_qid)) + '.features' 
         feature_diff = numpy.genfromtxt(feature_diff_file,delimiter=',')
         train_diff_features = numpy.vstack((train_diff_features,feature_diff))

         # getting the labels on train set from different annotators
         for annt_id in range(count_annts):
            cur_annt_labels = annt_labels[annt_id]
            annt_lables_for_qid_file = noisy_labels_dir + '/' + str(train_batch_qid) + '.noisy_labels' + str(annt_id+1)
            annt_lables_for_qid = numpy.genfromtxt(annt_lables_for_qid_file)
            cur_annt_labels = numpy.hstack((cur_annt_labels,annt_lables_for_qid))
            annt_labels[annt_id] = cur_annt_labels

      ext_diff_feats = numpy.hstack((train_diff_features,numpy.ones((train_diff_features.shape[0],1))))
      max_iter = 10
      TrainEM.TrainModel(ext_diff_features,annt_labels,max_iter) 
