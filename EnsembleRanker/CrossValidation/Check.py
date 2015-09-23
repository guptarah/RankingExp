"""
This script performs cv on EM algorithm that I have proposed
"""

import numpy
import scipy
import scipy.stats as stats
import sys
sys.path.append('/auto/rcf-proj/pg/guptarah/RankingExp/scripts/EnsembleRanker/MultiAnntScripts/')
sys.path.append('/auto/rcf-proj/pg/guptarah/RankingExp/scripts/EnsembleRanker/SVRanker/')
import TrainEM
import TrainEMRelEst
import SVRankerSoft  
import SVRanker3

def GetResults(w,features):
   ext_features = numpy.hstack((features,numpy.zeros((features.shape[0],1))))
   scores = numpy.dot(ext_features,w.T) 
   return numpy.mean(scores>0)

def PrintResults(w,features):
   ext_features = numpy.hstack((features,numpy.zeros((features.shape[0],1))))
   scores = numpy.dot(ext_features,w.T) 
   print 'Correctly identified pairs ratio:', numpy.mean(scores>0) 
   return numpy.mean(scores>0)

def PrintResultsStats(w,features,labels):
   ext_features = numpy.hstack((features,numpy.ones((features.shape[0],1))))
   scores = numpy.dot(ext_features,w.T) 
   print 'Kendall Tau: %f, Spearman correlation: %f, Pearson correlation: %f'\
      %(stats.kendalltau(scores,labels)[0], stats.spearmanr(scores,labels)[0], \
      numpy.corrcoef(scores.T,labels.T)[0,1])  
   return stats.spearmanr(scores,labels)[0]

def PerformCV(qid_file, diff_feat_dir, feat_file, true_labels_file, noisy_labels_dir, batch_size, count_annts):
#"""
#diff_feat_dir: directory where the diff features are stored
#noisy_labels_dir: directory where the noisy labels are stored
#true_labels_file: the true labels file used for test set evaluation
#qid_file: the qid file location
#noisy_labels_dir: directory containing noisy preferences corresponding to
#features in the diff_feat_dir
#count_annts: number of annotators
#
#Example values:
#qid_file= '/auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/qids'
#diff_feat_dir='/auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/noisy_features/labels/'
#feat_file='/auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/features'
#true_labels_file='/auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/labels'
#noisy_labels_dir='/auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/noisy_labels_pairwise'
#batch_size=1
#count_annts=6
#"""

   qids = numpy.genfromtxt(qid_file,dtype='int')
   qids_unique = numpy.unique(qids)

   features = numpy.genfromtxt(feat_file,delimiter=',')
   labels = numpy.genfromtxt(true_labels_file,delimiter=',')

   if numpy.remainder(len(qids_unique),batch_size):
      print "Please provide a split that divides number of unique qids"
      return

   num_batches = len(qids_unique)/batch_size
   mean_result_storage = numpy.zeros((4,5+count_annts))
   # 5 for True, EM, EMRelEst, Borda results and Majority vote and other for each annotator
   # 2 rows for correcit pairwise identification: first for test, second for dev
   # 2 rows for spearman correlation: first for test, second for dev
   
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
      test_diff_features = numpy.empty([0,test_features.shape[1]])
      for test_batch_qid in test_batch_qids:
         feature_diff_file = diff_feat_dir + '/labels/' + str(int(test_batch_qid)) + '.features' 
         feature_diff = numpy.genfromtxt(feature_diff_file,delimiter=',')
         test_diff_features = numpy.vstack((test_diff_features,feature_diff))

      dev_features = features[numpy.in1d(qids,dev_batch_qids),:] 
      dev_labels = numpy.matrix(labels[numpy.in1d(qids,dev_batch_qids)]).T
      dev_diff_features = numpy.empty([0,dev_features.shape[1]])
      for dev_batch_qid in dev_batch_qids:
         feature_diff_file = diff_feat_dir + '/labels/' + str(int(dev_batch_qid)) + '.features' 
         feature_diff = numpy.genfromtxt(feature_diff_file,delimiter=',')
         dev_diff_features = numpy.vstack((dev_diff_features,feature_diff))

      # get all train set features together
      train_diff_features = numpy.empty([0,test_features.shape[1]])
      annt_labels = numpy.empty([count_annts,0]).tolist()
      for train_batch_qid in train_batch_qids:
         feature_diff_file = diff_feat_dir + '/labels/' + str(int(train_batch_qid)) + '.features' 
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
      
      # Getting results using true labels during training
      for check_id in range(10):
         w = .01*numpy.ones((1,1+train_diff_features.shape[1]))
         print 'training true baseline model for iter ... %d' % (i)
         n_epochs, learning_rate, lambda_w = 2000, .01, .001
         random_scores = (numpy.random.uniform(0,1,size=(train_diff_features.shape[0],1)) > (.05*check_id))*1
         print 'correct supplied: ', numpy.mean(random_scores)
         w = SVRankerSoft.svr_optimization(train_diff_features,random_scores,w,learning_rate,n_epochs,lambda_w)

         print 'True Baseline Test Results:'
         mean_result_storage[2,0] += PrintResultsStats(w,test_features,test_labels)
         mean_result_storage[0,0] += PrintResults(w,test_diff_features) 

         print 'True Baseline Dev Results:'
         mean_result_storage[3,0] += PrintResultsStats(w,dev_features,dev_labels)
         mean_result_storage[1,0] += PrintResults(w,dev_diff_features) 

         print '-----------------------------'
         print ''


if __name__ == "__main__":
   if len(sys.argv) != 8:
      print 'Wrong usage:'
      print 'Correct usage: python PerformCV_EM.py \
      qid_file, diff_feat_dir, feat_file, true_labels_file, noisy_labels_dir, batch_size=1, count_annts'
      print 'Example Files:\
      qid_file= /auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/qids\
      diff_feat_dir= /auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/noisy_features/\
      feat_file= /auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/features\
      true_labels_file= /auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/labels\
      noisy_labels_dir= /auto/rcf-proj/pg/guptarah/RankingExp/data/wine_quality/noisy_labels_pairwise\
      batch_size= 1\
      count_annts= 6'
   else:
      qid_file = sys.argv[1]
      diff_feat_dir = sys.argv[2]
      feat_file = sys.argv[3]
      true_labels_file = sys.argv[4]
      noisy_labels_dir = sys.argv[5]
      batch_size = int(sys.argv[6])
      count_annts = int(sys.argv[7])      
      PerformCV(qid_file, diff_feat_dir, feat_file, true_labels_file, noisy_labels_dir, batch_size, count_annts)





 
