"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os 
import argparse
import sys
import numpy as np
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
import warnings
warnings.filterwarnings("ignore")


def main_metrics(ori_data, generated_data, args):
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  # 1. Discriminative Score
  discriminative_score = list()
  for _ in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(ori_data, generated_data)
    discriminative_score.append(temp_disc)
      
  metric_results['discriminative'] = np.mean(discriminative_score)
      
  # 2. Predictive score
  predictive_score = list()
  for tt in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(ori_data, generated_data)
    predictive_score.append(temp_pred)   
      
  metric_results['predictive'] = np.mean(predictive_score)     
          
  # 3. Visualization (PCA and tSNE)
  #visualization(ori_data, generated_data, 'pca')
  #visualization(ori_data, generated_data, 'tsne')
  
  ## Print discriminative and predictive scores
  print(metric_results)

  return metric_results


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_name',choices=['sines', 'ETTh' , 'stock' , 'energy'],default='ETTh',type=str)
  parser.add_argument('--gen_model',choices=['TimeGAN', 'Diffusion_TS' , 'TimeVAE'],default='Diffusion_TS', type=str)
  parser.add_argument('--num_training',default=10, type=int)
  parser.add_argument('--metric_iteration',help='iterations of the metric computation',default=10,type=int)

  args = parser.parse_args() 
  dataset_name = args.data_name
  if dataset_name == 'sines':    
      dataset_name_2 = 'sine'
  elif dataset_name == 'ETTh': 
      dataset_name_2 = 'etth'
  else : 
      dataset_name_2 = dataset_name
  gen_path = './{}/generated_data/{}/'.format(args.gen_model, dataset_name_2)
  list_predictive = []
  list_discriminative = []
  for training_id in range(args.num_training) : 

    # Calls main function  
    ori_data = np.load('./training_data/{}_ground_truth_24_train.npy'.format(dataset_name_2))
    ori_data = (ori_data - np.min(ori_data))/(np.max(ori_data) - np.min(ori_data))
    generated_data = np.load( os.path.join(gen_path , '{}_{}.npy'.format(dataset_name,training_id) ) )
    data_size = ori_data.shape[0]
    generated_data = generated_data[:data_size,:,:]
    generated_data = (generated_data - np.min(generated_data))/(np.max(generated_data) - np.min(generated_data))
    metric_results= main_metrics(ori_data, generated_data, args)
    list_predictive.append(metric_results['predictive'])
    list_discriminative.append(metric_results['discriminative'])

