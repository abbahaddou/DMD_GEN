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
from pydmd import DMD
import ot
import random
import warnings
warnings.filterwarnings("ignore")
def get_DMD_modes( x, svd_rank):
    modes = []
    for snapshots in x:
        dmd = DMD( svd_rank=svd_rank)
        dmd.fit(snapshots.T)
        modes.append(dmd.modes) # store by columns
    return np.array(modes)


def main_metrics(ori_data, generated_data, args):
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  # 1. Discriminative Score
  dmd_modes_score = list()
  dmd_modes_score_sin = list()
  data_size = ori_data.shape[0]
  num_samples = 128
  for _ in range(args.metric_iteration):
    ori_idx = random.sample(np.arange(data_size).tolist(), num_samples)
    gen_idx = random.sample(np.arange(data_size).tolist(), num_samples)
    sampled_ori = ori_data[ori_idx,:,:]
    sampled_gen = generated_data[gen_idx,:,:]
    M_theta = np.zeros([num_samples, num_samples])
    M_sin_theta = np.zeros([num_samples, num_samples])
    for i_ in range(num_samples) : 
      for j_ in range(num_samples) : 
          ori_i_ = sampled_ori[i_,:,:]
          gen_j_ = sampled_gen[j_,:,:]

          dmd1 = DMD(svd_rank=args.svd_rank)
          dmd1.fit(ori_i_.T.copy())
          dmd2 = DMD(svd_rank=args.svd_rank)
          dmd2.fit(gen_j_.T.copy())

          q1, r1 = np.linalg.qr(dmd1.modes)
          q2, r2 = np.linalg.qr(dmd2.modes)
          cos_theta = np.linalg.svd(q1.T @ q2, compute_uv=False)
          # cos_theta_diag = np.diag(cos_theta)
          theta_diag = np.arccos(cos_theta)
          theta_diag = np.nan_to_num(theta_diag, copy=True, nan=0.0) + 1e-15
          sin_theta_diag = np.sin(theta_diag)
          M_theta[i_,j_] = np.sqrt((theta_diag**2).sum() )
          M_sin_theta[i_,j_] = np.sqrt((sin_theta_diag**2).sum() )

    nx = ot.backend.get_backend(sampled_ori, sampled_gen)
    a = nx.from_numpy(ot.utils.unif(num_samples), type_as=sampled_gen)
    b = nx.from_numpy(ot.utils.unif(num_samples), type_as=sampled_gen)
    sinkhron_dist_theta = ot.bregman.sinkhorn2(a, b, M_theta, reg = 0.01)
    sinkhron_dist_sin_theta = ot.bregman.sinkhorn2(a, b, M_sin_theta, reg = 0.01)
    dmd_modes_score.append(sinkhron_dist_theta)
    dmd_modes_score_sin.append(sinkhron_dist_sin_theta)

  metric_results['dmd_dist_theta']  = np.mean(dmd_modes_score)
  metric_results['dmd_dist_sin_theta']  = np.mean(dmd_modes_score_sin)

  return metric_results


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_name',choices=['sines', 'ETTh' , 'stock' , 'energy'],default='sines',type=str)
  parser.add_argument('--gen_model',choices=['TimeGAN', 'Diffusion_TS' , 'TimeVAE'],default='Diffusion_TS', type=str)
  parser.add_argument('--num_training',default=10, type=int)
  parser.add_argument('--metric_iteration',help='iterations of the metric computation',default=10,type=int)

  parser.add_argument('--svd_rank', type= int,default = 4 )
  args = parser.parse_args() 
  dataset_name = args.data_name
  if dataset_name == 'sines':    
      dataset_name_2 = 'sine'
  elif dataset_name == 'ETTh': 
      dataset_name_2 = 'etth'
  else : 
      dataset_name_2 = dataset_name
  gen_path = '/home/yassine/PHD/Metrics/{}/generated_data/{}/'.format(args.gen_model, dataset_name_2)
  list_dmd_dist_theta = []
  list_dmd_dist_sin_theta = []
  for training_id in range(args.num_training) : 

    # Calls main function  
    ori_data = np.load('/home/yassine/PHD/TimeGAN/training_data/{}_ground_truth_24_train.npy'.format(dataset_name_2))

    generated_data = np.load( os.path.join(gen_path , '{}_{}.npy'.format(dataset_name,training_id) ) )
    data_size = ori_data.shape[0]
    generated_data = generated_data[:data_size,:,:]

    metric_results= main_metrics(ori_data, generated_data, args)
    list_dmd_dist_theta.append(metric_results['dmd_dist_theta'])
    list_dmd_dist_sin_theta.append(metric_results['dmd_dist_sin_theta'])



