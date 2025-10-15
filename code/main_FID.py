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
import wandb
import warnings
warnings.filterwarnings("ignore")
import torch
import scipy
import numpy as np

from ts2vec.ts2vec import TS2Vec


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def Context_FID(ori_data, generated_data):
    model = TS2Vec(input_dims=ori_data.shape[-1], device=torch.device('cuda'), batch_size=8, lr=0.001, output_dims=320,max_train_length=3000)

    model.fit(ori_data, verbose=False)
    ori_represenation = model.encode(ori_data, encoding_window='full_series')
    gen_represenation = model.encode(generated_data, encoding_window='full_series')
    idx = np.random.permutation(ori_data.shape[0])
    ori_represenation = ori_represenation[idx]
    gen_represenation = gen_represenation[idx]
    results = calculate_fid(ori_represenation, gen_represenation)
    return results


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
  parser.add_argument('--use_wandb', type= int,default = 0 )
  args = parser.parse_args() 
  dataset_name = args.data_name
  if dataset_name == 'sines':    
      dataset_name_2 = 'sine'
  elif dataset_name == 'ETTh': 
      dataset_name_2 = 'etth'
  else : 
      dataset_name_2 = dataset_name
  gen_path = '/home/yassine/PHD/Metrics/{}/generated_data/{}/'.format(args.gen_model, dataset_name_2)
  list_predictive = []
  list_discriminative = []
  for training_id in range(args.num_training) : 

    # Calls main function  
    ori_data = np.load('/home/yassine/PHD/TimeGAN/training_data/{}_ground_truth_24_train.npy'.format(dataset_name_2))
    generated_data = np.load( os.path.join(gen_path , '{}_{}.npy'.format(dataset_name,training_id) ) )
    data_size = ori_data.shape[0]
    generated_data = generated_data[:data_size,:,:]

    context_fid = Context_FID(ori_data, generated_data)
    print(context_fid)
    sys.exit()
    #list_predictive.append(metric_results['predictive'])

    #np.save(os.path.join(gen_path,   '{}_{}.npy'.format( dataset_name , training_id)), generated_data)
  if args.use_wandb == 1 : 
        wandb_api_key = "38121178c94cfeb04a598375b0ae4a3991bac385" # Do not share this notebook outside this scope !
        wandb_conx = wandb.login(key = wandb_api_key)
        print(f"Connected to Wandb online interface : {wandb_conx}")
        run = wandb.init(project='ICLR FID Score V2', 
                        id = "Dataset_{}_{}".format(args.gen_model,dataset_name) , 
                        name= "Dataset_{}_{}".format(args.gen_model,dataset_name), 
                        config={'model':args.gen_model  ,"Dataset":dataset_name})
        wandb.log({'mean_fid' : np.mean(list_predictive) , 'std_fid' : np.std(list_predictive) })
        wandb.finish()
 