#-*- coding: euc-kr -*-
#!/usr/bin/env python


import copy
import my_optim
from torch import distributed
import torch.distributed as dist
import argparse
import kaldi_io
import numpy as np
import torch
from torch.autograd import Variable
import timeit
import torch.optim as optim
import os
from data_io import load_chunk,load_counts,read_conf
import random
import torch.nn as nn
import sys
from torch.multiprocessing import Process
import time
from resource import getrusage as resource_usage, RUSAGE_SELF
from time import time as timestamp



class Parameter_Server:
  
  def __init__(self):
      self.checkpoint=0
  def ps_server(self,rank):
      os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
      torch.cuda.set_device(dist.get_rank())
      
      device = "cuda:{}".format(dist.get_rank())
      print("I am parameter server!")
      print("My rank is %d" % int(rank))
      a=0
      port = sys.argv[1]

      world_size = sys.argv[3]
      ip_add = sys.argv[4]
     
      options=read_conf()
      
      do_training=bool(int(options.do_training))
      do_eval=bool(int(options.do_eval))
      do_forward=bool(int(options.do_forward))
     
      fea_scp=options.fea_scp
      fea_opts=options.fea_opts
      lab_folder=options.lab_folder
      lab_opts=options.lab_opts
      
      dev_fea_scp="/home/slave3/kaldi/egs/timit/s5/pytorch-kaldi/exp/mfcc_shu/dev_split.000"
      dev_fea_opts="apply-cmvn --utt2spk=ark:$KALDI_ROOT/egs/timit/s5/data/dev/utt2spk  ark:$PYTORCH_EXP/mfcc_shu/dev_cmvn_speaker.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |"
      dev_lab_folder='/home/slave3/kaldi/egs/timit/s5/exp/dnn4_pretrain-dbn_dnn_ali_dev'
      dev_lab_opts='ali-to-pdf'
      
      
      
      out_file=options.out_file
      

      count_file=options.count_file
      pt_file=options.pt_file
      

      left=int(options.cw_left)
      right=int(options.cw_right)
      seed=int(options.seed)
      use_cuda=bool(int(options.use_cuda))
      multi_gpu=bool(int(options.multi_gpu))
      NN_type=options.NN_type

      batch_size=int(options.batch_size)
      lr=float(options.lr)
      save_gpumem=int(options.save_gpumem)
      opt=options.optimizer
      if NN_type=='RNN':
         from neural_nets import RNN as ann
         rnn=1
      
      if NN_type=='LSTM':
         from neural_nets import LSTM as ann
         rnn=1
         
      if NN_type=='GRU':
        from neural_nets import GRU as ann
        rnn=1
      if NN_type=='MLP':
         from neural_nets import MLP as ann
         rnn=0
      options.input_dim=429

      options.num_classes=1944
      print(options.input_dim)
      print(options.num_classes)
      
      sh_model = ann(options)

      optimizer = optim.SGD(sh_model.parameters(), lr=lr) 

      sh_model.cuda(device=device) 
      check = 0
      
      for shared_param in sh_model.parameters():
        check=check+1
        if check==1:
          shared_param.grad = torch.zeros([1024,429]).cuda(device=device)
        elif check==2:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==3:
          shared_param.grad = torch.zeros([1024,1024]).cuda(device=device)
        elif check==4:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==5:
          shared_param.grad = torch.zeros([1024,1024]).cuda(device=device)
        elif check==6:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==7:
          shared_param.grad = torch.zeros([1024,1024]).cuda(device=device)
        elif check==8:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==9:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==10:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==11:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==12:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==13:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==14:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==15:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==16:
          shared_param.grad = torch.zeros([1024]).cuda(device=device)
        elif check==17:
          shared_param.grad = torch.zeros([1944,1024]).cuda(device=device)
        elif check==18:
          shared_param.grad = torch.zeros([1944]).cuda(device=device)

      
      print("parameter server initialize")
      FLAG = torch.zeros(1)
      while True:

        target_recv = dist.recv(tensor=FLAG)
        
        if FLAG==3:
          for shared_param in sh_model.parameters():
               
            if target_recv == 1:
                    dist.recv(tensor=shared_param._grad, src=1)
                    
        
            elif target_recv == 2:
                    dist.recv(tensor=shared_param._grad, src=2)
                    
  
            elif target_recv == 3:
                    dist.recv(tensor=shared_param._grad, src=3)
                    
                      
            elif target_recv == 4:
                    dist.recv(tensor=shared_param._grad, src=4)
                  
          optimizer.step()
          for param in sh_model.parameters():
             
              if target_recv ==1:
                         
                 dist.send(tensor=param.data, dst=1)
                  
              elif target_recv ==2:
                
                   
                 dist.send(tensor=param.data, dst=2)
                  
              elif target_recv ==3:
                
                   
                 dist.send(tensor=param.data, dst=3)
                  
              elif target_recv ==4:
                
                 dist.send(tensor=param.data, dst=4)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                  

            
            
            
            