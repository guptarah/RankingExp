import numpy
import sys
from KendallTau import Kendall 
from SpearmanCorr import CORRELATION 
import os

def AddNoise(labels_file,k=0):
   labels = numpy.genfromtxt(labels_file,delimiter=',',dtype='int')
   max_label = numpy.amax(labels)
   min_label = numpy.amin(labels)
   print max_label, min_label  
 
   size_labels = labels.shape
   noise1 = (1*k)*numpy.random.uniform(-1,1,size_labels) # usually this is 1 + .5 everytime 
   noise2 = (1.5*k)*numpy.random.uniform(-1,1,size_labels) 
   noise3 = (2*k)*numpy.random.uniform(-1,1,size_labels) 
   noise4 = (2.5*k)*numpy.random.uniform(-1,1,size_labels) 
   noise5 = (3*k)*numpy.random.uniform(-1,1,size_labels) 
   noise6 = (3.5*k)*numpy.random.uniform(-1,1,size_labels) 
   noise7 = (4*k)*numpy.random.uniform(-1,1,size_labels) 
   noise8 = (4.5*k)*numpy.random.uniform(-1,1,size_labels) 
   noise9 = (4.5*k)*numpy.random.uniform(-1,1,size_labels) 
   
   noisy_labels1 = numpy.around((noise1 + labels))
   noisy_labels1[noisy_labels1<min_label] = min_label
   noisy_labels1[noisy_labels1>max_label] = max_label 
   noisy_labels1 = noisy_labels1.astype(int) 
   
   noisy_labels2 = numpy.around((noise2 + labels))
   noisy_labels2[noisy_labels2<min_label] = min_label
   noisy_labels2[noisy_labels2>max_label] = max_label 
   
   noisy_labels3 = numpy.around((noise3 + labels))
   noisy_labels3[noisy_labels3<min_label] = min_label
   noisy_labels3[noisy_labels3>max_label] = max_label 
   
   noisy_labels4 = numpy.around((noise4 + labels))
   noisy_labels4[noisy_labels4<min_label] = min_label
   noisy_labels4[noisy_labels4>max_label] = max_label 
   
   noisy_labels5 = numpy.around((noise5 + labels))
   noisy_labels5[noisy_labels5<min_label] = min_label
   noisy_labels5[noisy_labels5>max_label] = max_label 
   
   noisy_labels6 = numpy.around((noise6 + labels))
   noisy_labels6[noisy_labels6<min_label] = min_label
   noisy_labels6[noisy_labels6>max_label] = max_label 
  
   noisy_labels7 = numpy.around((noise7 + labels))
   noisy_labels7[noisy_labels7<min_label] = min_label
   noisy_labels7[noisy_labels7>max_label] = max_label 
   
   noisy_labels8 = numpy.around((noise8 + labels))
   noisy_labels8[noisy_labels8<min_label] = min_label
   noisy_labels8[noisy_labels8>max_label] = max_label 
   
   noisy_labels9 = numpy.around((noise9 + labels))
   noisy_labels9[noisy_labels9<min_label] = min_label
   noisy_labels9[noisy_labels9>max_label] = max_label 
   
   save_dir = os.path.dirname(os.path.abspath(labels_file))
   print save_dir
 
   numpy.savetxt(save_dir+'/noisy_labels1', noisy_labels1, fmt='%f', delimiter=' ')
   numpy.savetxt(save_dir+'/noisy_labels2', noisy_labels2, fmt='%f', delimiter=' ')
   numpy.savetxt(save_dir+'/noisy_labels3', noisy_labels3, fmt='%f', delimiter=' ')
   numpy.savetxt(save_dir+'/noisy_labels4', noisy_labels4, fmt='%f', delimiter=' ')
   numpy.savetxt(save_dir+'/noisy_labels5', noisy_labels5, fmt='%f', delimiter=' ')
   numpy.savetxt(save_dir+'/noisy_labels6', noisy_labels6, fmt='%f', delimiter=' ')
   numpy.savetxt(save_dir+'/noisy_labels7', noisy_labels7, fmt='%f', delimiter=' ')
   numpy.savetxt(save_dir+'/noisy_labels8', noisy_labels8, fmt='%f', delimiter=' ')
   numpy.savetxt(save_dir+'/noisy_labels9', noisy_labels9, fmt='%f', delimiter=' ')
   
   print Kendall(labels, noisy_labels1)
   print Kendall(labels, noisy_labels2)
   print Kendall(labels, noisy_labels3)
   print Kendall(labels, noisy_labels4)
   print Kendall(labels, noisy_labels5)
   print Kendall(labels, noisy_labels6)
  
   borda_count1 = labels + (noise1 + noise2 + noise3 + noise4 + noise5 + noise6)/6
   borda_count2 = (noisy_labels1 + noisy_labels2 + noisy_labels3 + noisy_labels4 + noisy_labels5 + noisy_labels6)/6
   print Kendall(labels, borda_count1)
   print Kendall(labels, borda_count2)

if __name__ == "__main__":
   if len(sys.argv) != 3:
      print 'please give the labels file and noise level\
             Usage: python AddNoiseToLabel.py LablesFile noise_level' 
   else:
      labels_file = sys.argv[1] 
      noise_level = float(sys.argv[2])
      AddNoise(labels_file,noise_level)
