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

      print annt_labels[0].shape

      for annt_id in range(count_annts):
         annt_labels[annt_id] = numpy.ravel(numpy.random.uniform(0,1,size=(train_diff_features.shape[0],1)) > .3)*1

      print annt_labels[0].shape

      ext_diff_feats = numpy.hstack((train_diff_features,numpy.ones((train_diff_features.shape[0],1))))
      max_iter = 20
      w,k = TrainEM.TrainModel(ext_diff_feats,annt_labels,max_iter)
      print numpy.mean(k>0.5)
      w = .01*numpy.ones((1,1+train_diff_features.shape[1]))
      n_epochs, learning_rate, lambda_w = 2800, .01, .001

      last_dev_result = 0
      for train_epoch in range(n_epochs): 
         w = SVRankerSoft.svr_optimization(train_diff_features,1*(k>0.5),w,.01,1,.001)      
         cur_dev_result = GetResults(w,dev_diff_features)
         delta_performance = cur_dev_result - last_dev_result
         if (delta_performance < .0000) and (cur_dev_result > last_dev_result):
            print 'break at iter ',train_epoch
            break 
         last_dev_result = cur_dev_result

      print 'Basic multiple annotator Test Results:'
      mean_result_storage[2,1] += PrintResultsStats(w,test_features,test_labels)
      mean_result_storage[0,1] += PrintResults(w,test_diff_features) 
 
      print 'Basic multiple annotator Dev Results:'
      mean_result_storage[3,1] += PrintResultsStats(w,dev_features,dev_labels)
      mean_result_storage[1,1] += PrintResults(w,dev_diff_features) 

      print 'Correct identifications on train set: %f' %(numpy.mean(k>.5))

      print '-----------------------------'
      print ''

#      # Training model using TrainEMRelEst function
#      w,k = TrainEMRelEst.TrainModel(ext_diff_feats,annt_labels,max_iter)
#      w = SVRankerSoft.svr_optimization(train_diff_features,numpy.around(k),w,.02,2000,.001)      
#      
#      print 'Rel est multiple annotator Test Results:'
#      mean_result_storage[2,2] += PrintResultsStats(w,test_features,test_labels)
#      mean_result_storage[0,2] += PrintResults(w,test_diff_features) 
# 
#      print 'Rel est multiple annotator Dev Results:'
#      mean_result_storage[3,2] += PrintResultsStats(w,dev_features,dev_labels)
#      mean_result_storage[1,2] += PrintResults(w,dev_diff_features) 
#
#      print 'Correct identifications on train set: %f' %(numpy.mean(k>.5))
#
#      print '-----------------------------'
#      print ''

      # Getting results using majority vote 
      majority_vote = ((numpy.mean(numpy.matrix(annt_labels),axis=0) > .5)*1).T
      w = .01*numpy.ones((1,1+train_diff_features.shape[1]))
      #w = SVRankerSoft.svr_optimization(train_diff_features,majority_vote,w,.02,2000,.001)      
      
      last_dev_result = 0
      for train_epoch in range(n_epochs): 
         w = SVRankerSoft.svr_optimization(train_diff_features,majority_vote,w,.01,1,.001)      
         cur_dev_result = GetResults(w,dev_diff_features)
         delta_performance = cur_dev_result - last_dev_result
         if (delta_performance < .0000) and (cur_dev_result > last_dev_result):
            print 'break at iter ',train_epoch
            break 
         last_dev_result = cur_dev_result
      print 'Majority vote Test Results:'
      
      mean_result_storage[2,4] += PrintResultsStats(w,test_features,test_labels)
      mean_result_storage[0,4] += PrintResults(w,test_diff_features) 
  
      print 'Majority vote Dev Results:'
      mean_result_storage[3,4] += PrintResultsStats(w,dev_features,dev_labels)
      mean_result_storage[1,4] += PrintResults(w,dev_diff_features) 
 
      print 'Correct identifications on train set: %f' %(numpy.mean(majority_vote))
      print '-----------------------------'
      print ''
      
      # Getting results using true labels during training
      w = .01*numpy.ones((1,1+train_diff_features.shape[1]))
      print 'training true baseline model for iter ... %d' % (i)
      n_epochs, learning_rate, lambda_w = 2000, .01, .001
      #w = SVRanker3.svr_optimization(train_diff_features,w,learning_rate,n_epochs,lambda_w)
      w = SVRankerSoft.svr_optimization(train_diff_features,numpy.ones(majority_vote.shape),w,learning_rate,n_epochs,lambda_w)
   
      print 'True Baseline Test Results:'
      mean_result_storage[2,0] += PrintResultsStats(w,test_features,test_labels)
      mean_result_storage[0,0] += PrintResults(w,test_diff_features) 
  
      print 'True Baseline Dev Results:'
      mean_result_storage[3,0] += PrintResultsStats(w,dev_features,dev_labels)
      mean_result_storage[1,0] += PrintResults(w,dev_diff_features) 
 
      print '-----------------------------'
      print ''

      print 'Getting results for each annotator'
      # Getting results on each annotator
      w_borda = numpy.zeros((1,1+train_diff_features.shape[1]))
      for noisy_annt_id in range(count_annts):
         print 'at noisy annotor id: %d' %(noisy_annt_id)
#         train_diff_features = numpy.empty([0,test_features.shape[1]])
#         for train_batch_qid in train_batch_qids:
#            feature_diff_file = diff_feat_dir + '/noisy_labels' + str(noisy_annt_id+1) + '/' + str(int(train_batch_qid)) + '.features'
#            feature_diff = numpy.genfromtxt(feature_diff_file,delimiter=',')
#            train_diff_features = numpy.vstack((train_diff_features,feature_diff)) 
         cur_annt_labels = numpy.matrix(annt_labels[noisy_annt_id]).T
         w = .01*numpy.ones((1,1+train_diff_features.shape[1])) # initial w
         w = SVRankerSoft.svr_optimization(train_diff_features,cur_annt_labels,w,learning_rate,n_epochs,lambda_w)
  
         print 'Annotator %d Test Results:' %(noisy_annt_id)
         mean_result_storage[2,5+noisy_annt_id] += PrintResultsStats(w,test_features,test_labels)
         mean_result_storage[0,5+noisy_annt_id] += PrintResults(w,test_diff_features) 
  
         print 'Annotator %d Dev Results:' %(noisy_annt_id)
         mean_result_storage[3,5+noisy_annt_id] += PrintResultsStats(w,dev_features,dev_labels)
         mean_result_storage[1,5+noisy_annt_id] += PrintResults(w,dev_diff_features) 

         w_borda = w_borda + w/numpy.linalg.norm(w,2) 
         print 'Correct identifications on train set: %f' %(numpy.mean(cur_annt_labels))

      print 'Borda count Test Results:' 
      mean_result_storage[2,3] += PrintResultsStats(w_borda,test_features,test_labels)
      mean_result_storage[0,3] += PrintResults(w,test_diff_features) 
  
      print 'Borda count Dev Results:' 
      mean_result_storage[3,3] += PrintResultsStats(w_borda,dev_features,dev_labels)
      mean_result_storage[1,3] += PrintResults(w,dev_diff_features) 


      print 'Running mean of performances:'  
      print mean_result_storage 


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





 
