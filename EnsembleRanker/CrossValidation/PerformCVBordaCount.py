"""
This script takes in data splits, feature files, diff feature files, splits in each fold and MULTIPLE LABEL FILES
You get results by training on noisy labels when compared with the true labels.
Plain unweigted summation mergining is going on based on Borda Count. 
"""

import numpy
import scipy.stats as stats
import sys
sys.path.append('/auto/rcf-proj/pg/guptarah/RankingExp/scripts/EnsembleRanker/SVRanker')
sys.path.append('/auto/rcf-proj/pg/guptarah/RankingExp/scripts/DataPreparation/NoisyDataCreation')
import SVRanker3

def PerformCV(qid_file,diff_feat_dir,feat_file,labels_file_dir,cnt_noisy_labels_file,batch_size=5):
   """
   Given the list of qids, it takes the training features from diff_feat folder per noisy annotator,
   test and dev features from feat_file.
   Model is trained on diff_feat_dir files and evaluated on feat_file features for Kendall tau

   Make sure that the sum of split ratio is a divisor of number of unique qids
   """

   qids = numpy.genfromtxt(qid_file)
   qids_unique = numpy.unique(qids)

   features = numpy.genfromtxt(feat_file,delimiter=',')

   # creating labels list
   labels_file = labels_file_dir + '/labels'
   true_labels = numpy.genfromtxt(labels_file,delimiter=',') 

   if numpy.remainder(len(qids_unique),batch_size):
      print "Please provide a split that divides number of unique qids"
      return

   num_batches = len(qids_unique)/batch_size

   all_test_scores, all_dev_scores = numpy.empty([0,1]), numpy.empty([0,1])
   all_test_labels, all_dev_labels = numpy.empty([0,1]), numpy.empty([0,1])
   for i in range(num_batches):
      # Determine the qids in test, dev and train sets
      test_id = i 
      test_batch_qids = qids_unique[numpy.arange(batch_size*test_id,batch_size*(test_id+1))]

      dev_id = numpy.remainder(i+1,num_batches)
      dev_batch_qids = qids_unique[numpy.arange(batch_size*dev_id,batch_size*(dev_id+1))]
      
      train_batch_qids = numpy.setdiff1d(qids_unique,numpy.union1d(test_batch_qids,dev_batch_qids))  
   
      # find the features and labels for the train and the dev set
      test_features = features[numpy.in1d(qids,test_batch_qids).T,:] 
      test_labels = numpy.matrix(true_labels[numpy.in1d(qids,test_batch_qids)]).T
      dev_features = features[numpy.in1d(qids,dev_batch_qids),:] 
      dev_labels = numpy.matrix(true_labels[numpy.in1d(qids,dev_batch_qids)]).T

      # get all train set features together from all the annotators
      w_per_annt = numpy.empty((cnt_noisy_labels_file,0)).tolist() 
      for noisy_annt_id in range(cnt_noisy_labels_file):
         print 'at noisy annotor id: %d' %(noisy_annt_id)
         train_diff_features = numpy.empty([0,test_features.shape[1]]) 
         for train_batch_qid in train_batch_qids:
            feature_diff_file = diff_feat_dir + '/noisy_labels' + str(noisy_annt_id+1) + '/' + str(int(train_batch_qid)) + '.features' 
            feature_diff = numpy.genfromtxt(feature_diff_file,delimiter=',')
            train_diff_features = numpy.vstack((train_diff_features,feature_diff))
      
         w = numpy.ones((1,1+train_diff_features.shape[1])) # initial w
         print 'training model for iter ... %d' % (i)
         for iter in range(1):
            n_epochs, learning_rate, lambda_w = 2400, .02, .001
            w = SVRanker3.svr_optimization(train_diff_features,w,learning_rate,n_epochs,lambda_w)		

         w_per_annt[noisy_annt_id] = w
   
      # perform unweighted fusion to get results on test and dev set
      test_features_ext = numpy.hstack((test_features,numpy.ones((test_features.shape[0],1))))
      test_scores = 0
      for noisy_annt_id in range[cnt_noisy_labels_file]:
         test_scores += numpy.dot(test_features_ext,w_per_annt[noisy_annt_id].T)

      dev_features_ext = numpy.hstack((dev_features,numpy.ones((dev_features.shape[0],1))))
      dev_scores = 0
      for noisy_annt_id in range[cnt_noisy_labels_file]:
         dev_scores += numpy.dot(dev_features_ext,w_per_annt[noisy_annt_id].T)
         
      print 'Model Trained.'
      print 'Results:'
      print 'Iter Test set results:'
      print 'Kendall Tau:', stats.kendalltau(test_scores,test_labels)[0]
      print 'Spearman correlation:', stats.spearmanr(test_scores,test_labels)[0]
      print 'Pearson correlation:', numpy.corrcoef(test_scores.T,test_labels.T)[0,1]

      print 'Iter Dev set results:'
      print 'Kendall Tau:', stats.kendalltau(dev_scores[:,0],dev_labels[:,0])[0]
      print 'Spearman correlation:', stats.spearmanr(dev_scores,dev_labels)[0]
      print 'Pearson correlation:', numpy.corrcoef(dev_scores.T,dev_labels.T)[0,1]
         

      all_test_scores = numpy.vstack((all_test_scores,test_scores))
      all_test_labels = numpy.vstack((all_test_labels,test_labels))
      all_dev_scores = numpy.vstack((all_dev_scores,dev_scores))
      all_dev_labels = numpy.vstack((all_dev_labels,dev_labels))

      print 'Test set results:'
      print 'Kendall Tau:', stats.kendalltau(all_test_scores[:,0],all_test_labels[:,0])[0]
      print 'Spearman correlation:', stats.spearmanr(all_test_scores,all_test_labels)[0]
      print 'Pearson correlation:', numpy.corrcoef(all_test_scores.T,all_test_labels.T)[0,1]

      print 'Dev set results:'
      print 'Kendall Tau:', stats.kendalltau(all_dev_scores[:,0],all_dev_labels[:,0])[0]
      print 'Spearman correlation:', stats.spearmanr(all_dev_scores,all_dev_labels)[0]
      print 'Pearson correlation:', numpy.corrcoef(all_dev_scores.T,all_dev_labels.T)[0,1]
      print '-------------------'
      print ''

if __name__ == "__main__":
   if len(sys.argv) != 6:
      print "Incorrect Usage \
      Usage: python PerformCV.py qid_file diff_feat_dir feat_file labels_file batch_size"
   else:
      qid_file = sys.argv[1]
      diff_feat_dir = sys.argv[2] # directory containing features determined
      # based on noisy labels. Ex:
      # /auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/noisy_features
      feat_file = sys.argv[3]
      labels_file_dir = sys.argv[4] # directory containing labels file as well
      # as noisy labels file [labels, noisy_labels1, noisy_labels2, ... ]
      cnt_noisy_labels_file = int(sys.argv[5])
      batch_size = int(sys.argv[6])
      PerformCV(qid_file,diff_feat_dir,feat_file,labels_file,cnt_noisy_labels_file,batch_size)
