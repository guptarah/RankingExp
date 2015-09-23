#├── NoisyDataCreation
#│   ├── AddNoiseToLabel.py : First run this to get noisy annotators
#├── PairwiseFeatureCreation
#│   ├── PreparePairwiseFeatures.py : Then run this to get difference features based on noisy labels
#│   └── PreparePairwiseFeaturesWNoisyLabels.py : Then run this to get difference features based on 
#                                     golden labels as well as noisy comparison labels by annotators

input_dir=$1
noise_level=$2

python DataPreparation/NoisyDataCreation/AddNoiseToLabel.py $input_dir/labels $noise_level 
python DataPreparation/PairwiseFeatureCreation/PreparePairwiseFeatures.py $input_dir/noisy_labels1 $input_dir/features $input_dir/qids
python DataPreparation/PairwiseFeatureCreation/PreparePairwiseFeatures.py $input_dir/noisy_labels2 $input_dir/features $input_dir/qids
python DataPreparation/PairwiseFeatureCreation/PreparePairwiseFeatures.py $input_dir/noisy_labels3 $input_dir/features $input_dir/qids
python DataPreparation/PairwiseFeatureCreation/PreparePairwiseFeatures.py $input_dir/noisy_labels4 $input_dir/features $input_dir/qids
python DataPreparation/PairwiseFeatureCreation/PreparePairwiseFeatures.py $input_dir/noisy_labels5 $input_dir/features $input_dir/qids
python DataPreparation/PairwiseFeatureCreation/PreparePairwiseFeatures.py $input_dir/noisy_labels6 $input_dir/features $input_dir/qids
python DataPreparation/PairwiseFeatureCreation/PreparePairwiseFeatures.py $input_dir/noisy_labels7 $input_dir/features $input_dir/qids
python DataPreparation/PairwiseFeatureCreation/PreparePairwiseFeatures.py $input_dir/noisy_labels8 $input_dir/features $input_dir/qids
python DataPreparation/PairwiseFeatureCreation/PreparePairwiseFeatures.py $input_dir/noisy_labels9 $input_dir/features $input_dir/qids
python DataPreparation/PairwiseFeatureCreation/PreparePairwiseFeaturesWNoisyLabels.py $input_dir/labels $input_dir 9 $input_dir/features $input_dir/qids

#Training models
#on all the data 
#9. python TrainOnAll.py ../../../data/wine_quality/Noise3/qids ../../../data/wine_quality/Noise3/noisy_features/ ../../../data/wine_quality/Noise3/features  ../../../data/wine_quality/Noise3/labels ../../../data/wine_quality/Noise3/noisy_labels_pairwise/ 1 6

#using cross validation
#10. python PerformCV_All.py ../../../data/wine_quality/Noise3/qids ../../../data/wine_quality/Noise3/noisy_features/ ../../../data/wine_quality/Noise3/features  ../../../data/wine_quality/Noise3/labels ../../../data/wine_quality/Noise3/noisy_labels_pairwise/ 1 6 
