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
from ps_server import Parameter_Server


class MAIN_CLASS:
  def __init__(self):
      self.check=0
      self.FLAG = torch.zeros(1)
  def ensure_shared_params(self,net,rank):    

      self.FLAG += 3
      dist.send(tensor=self.FLAG, dst=0)
      for param in net.parameters():
         
          dist.send(tensor=param.grad.data, dst=0)
      self.FLAG -= 3 
      for param in net.parameters():
      
          dist.recv(tensor=param.data, src=0)

  
  
  def main(self,rank):
      os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
      options=read_conf()

      do_training=bool(int(options.do_training))
      do_eval=bool(int(options.do_eval))
      do_forward=bool(int(options.do_forward))
      if do_forward:
        torch.cuda.set_device(0)
        device = "cuda:{}".format(0)
      else:
        torch.cuda.set_device(dist.get_rank()-1)
        device = "cuda:{}".format(dist.get_rank()-1)
      PS = Parameter_Server()
      if int(rank)==0 and do_training:
        PS.ps_server(rank)
      port = sys.argv[1]
      world_size = sys.argv[3]
      ip_add = sys.argv[4]


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

      net = ann(options)
      if use_cuda:
            net.cuda(device=device)
      update_time=0
      sum_update_time=0
      st_update_time=0
      end_update_time=0
      
      
      shu_time=0
      sum_shu_time=0
      st_shu_time=0
      end_shu_time=0
      
      model_time=0
      sum_model_time=0
      st_model_time=0
      end_model_time=0
      
      load_time=0
      sum_load_time=0
      st_load_time=0
      end_load_time=0
      
      val_time=0
      sum_val_time=0
      st_val_time=0
      end_val_time=0
      
      epoch_time=0
      sum_epoch_time=0
      st_epoch_time=0
      end_epoch_time=0  
      
      data_time=0

      st_data_time=0
      end_data_time=0 
      
      
      train_time=0

      st_train_time=0
      end_train_time=0 
      _, st_train_time= timestamp(), resource_usage(RUSAGE_SELF)   

      torch.manual_seed(seed)
      random.seed(seed)
      print("[INFO] Batch size: ",batch_size)
      if rnn or do_eval or do_forward:
         seed=-1
      _, st_data_time= timestamp(), resource_usage(RUSAGE_SELF)   
      if do_forward == 1:
        dev_data_name=[0]
      if do_forward == 0:
        [dev_data_name,dev_data_set_ori,dev_data_end_index]=load_chunk(dev_fea_scp,dev_fea_opts,dev_lab_folder,dev_lab_opts,left,right,-1)   

      [data_name,data_set_ori,data_end_index]=load_chunk(fea_scp,fea_opts,lab_folder,lab_opts,left,right,seed)

      data_len = int(len(data_set_ori)/(int(world_size)-1))
      if do_training:
        if int(world_size)-1==1:
          print("Partition data 1")
        elif int(world_size)-1==2:
          print("partition data 2")
          if int(rank)==1:
            data_set_ori = data_set_ori[0:data_len]
          elif int(rank)==2:
            data_set_ori = data_set_ori[data_len:]
        elif int(world_size)-1==3:
          print("partition data 3")
          if int(rank)==1:
            data_set_ori = data_set_ori[0:data_len]
          elif int(rank)==2:
            data_set_ori = data_set_ori[data_len:data_len*2]
          elif int(rank)==3:
            data_set_ori = data_set_ori[data_len*2:]
        elif int(world_size)-1==4:
          print("partition data 4")
          if int(rank)==1:
            data_set_ori = data_set_ori[0:data_len]
          elif int(rank)==2:
            data_set_ori = data_set_ori[data_len:data_len*2]
          elif int(rank)==3:
            data_set_ori = data_set_ori[data_len*2:data_len*3]
          elif int(rank)==4:
            data_set_ori = data_set_ori[data_len*3:]
        data_len = len(data_set_ori)

      end_data_time,_  = resource_usage(RUSAGE_SELF), timestamp()
      data_time = end_data_time.ru_utime - st_data_time.ru_utime
      print("data generate time: ", data_time)


      print(np.shape(data_set_ori))

      if not(save_gpumem):
         data_set=torch.from_numpy(data_set_ori).float().cuda(device=device)
      else:
         data_set=torch.from_numpy(data_set_ori).float()   
      if do_forward ==0:  
        if not(save_gpumem):
           dev_data_set=torch.from_numpy(dev_data_set_ori).float().cuda(device=device)
        else:
           dev_data_set=torch.from_numpy(dev_data_set_ori).float()  

      N_fea=data_set.shape[1]-1
      options.input_dim=N_fea
      N_out=int(data_set[:,N_fea].max()-data_set[:,N_fea].min()+1) 
      options.num_classes=N_out
      

      if multi_gpu:
       net = nn.DataParallel(net)
       
       

      
      optimizer_worker=None       

      if optimizer_worker is None:
              optimizer_worker = optim.SGD(net.parameters(), lr=lr)
      else:
        optimizer_worker = optim.RMSprop(net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
      if do_forward:     
        if pt_file!='none':
          checkpoint_load = torch.load(pt_file)
          net.load_state_dict(checkpoint_load['model_par'])
          optimizer_worker.load_state_dict(checkpoint_load['optimizer_par'])
          optimizer_worker.param_groups[0]['lr']=lr

      dev_N_snt=len(dev_data_name)
      N_snt=len(data_name)
      
      
      if do_training:
        print("do training")
        net.train()
        test_flag=0   

        if do_training:
          N_batches=int((N_snt/batch_size)/(int(world_size)-1))
        else:
          N_batches=int(N_snt/batch_size) 
 
        if rnn==0:
         N_ex_tr=data_set.shape[0]
         N_batches=int(N_ex_tr/batch_size)
         
      if do_eval:
       N_batches=N_snt
       net.eval()
       test_flag=1
       batch_size=1
       
       if do_forward:
        post_file=kaldi_io.open_or_fd(out_file,'wb')
        counts = load_counts(count_file)
        

      beg_batch=0
      end_batch=beg_batch+batch_size   
      
      dev_beg_batch=0
      dev_end_batch=dev_beg_batch+1
      
      
      snt_index=0
      beg_snt=0 
      dev_beg_snt=0
      loss_sum=0
      err_sum=0
      dev_loss_sum=0
      dev_err_sum=0
      temp_err=0
      dev_err_sum_tot=0
      dev_N_batches=0

      num_epoch=24
      main_class = MAIN_CLASS()
      if do_forward:
        for i in range(N_batches):
           if do_training :
            
            if rnn==1:
             max_len=data_end_index[snt_index+batch_size-1]-data_end_index[snt_index+batch_size-2]
           
             inp= Variable(torch.zeros(max_len,batch_size,N_fea)).contiguous()
             lab= Variable(torch.zeros(max_len,batch_size)).contiguous().long()
             
           
             for k in range(batch_size):
              snt_len=data_end_index[snt_index]-beg_snt
              N_zeros=max_len-snt_len
              N_zeros_left=random.randint(0,N_zeros)
              inp[N_zeros_left:N_zeros_left+snt_len,k,:]=data_set[beg_snt:beg_snt+snt_len,0:N_fea] 
              lab[N_zeros_left:N_zeros_left+snt_len,k]=data_set[beg_snt:beg_snt+snt_len,-1]
              
              beg_snt=data_end_index[snt_index]
              snt_index=snt_index+1
           
            else: 

             inp= Variable(data_set[beg_batch:end_batch,0:N_fea]).contiguous().cuda(device=device)
             lab= Variable(data_set[beg_batch:end_batch,N_fea]).contiguous().long().cuda(device=device)
             
            
           if do_eval:
              end_snt=data_end_index[i]
              inp= Variable(data_set[beg_snt:end_snt,0:N_fea],volatile=True).contiguous().cuda(device=device)
              lab= Variable(data_set[beg_snt:end_snt,N_fea],volatile=True).contiguous().long().cuda(device=device)
              if rnn==1:
                inp=inp.view(inp.shape[0],1,inp.shape[1])
                lab=lab.view(lab.shape[0],1)
              beg_snt=data_end_index[i]
            
           
           [loss,err,pout] = net(inp,lab,test_flag,rank)
           
           if multi_gpu:
             loss=loss.mean()
             err=err.mean()
        
           if do_forward:
            if rnn==1:
               pout=pout.view(pout.shape[0]*pout.shape[1],pout.shape[2]) 
            if int(rank)==0:
              kaldi_io.write_mat(post_file, pout.data.cpu().numpy()-np.log(counts/np.sum(counts)), data_name[i])
            
           if do_training:

            optimizer.zero_grad()  
          

            loss.backward()


            optimizer.step()

           
           loss_sum=loss_sum+loss.data
           err_sum=err_sum+err.data

           beg_batch=end_batch
           end_batch=beg_batch+batch_size

      else:

       m=0 
       for e in range(num_epoch):
        print("Batch size: ",m)
        _, st_epoch_time= timestamp(), resource_usage(RUSAGE_SELF)
        if e>0:
          
          dev_N_batches=dev_N_snt
          if e>1:
              temp_err=dev_err_sum_tot

          net.eval()
          test_flag=1
          dev_batch_size=1
          dev_beg_batch=0
          dev_end_batch=dev_beg_batch+1
          dev_loss_sum=0
          dev_err_sum=0
          dev_beg_snt=0
          _, st_val_time= timestamp(), resource_usage(RUSAGE_SELF)
          
          
          for j in range(dev_N_batches):
               
                end_snt=dev_data_end_index[j]
                dev_inp= Variable(dev_data_set[dev_beg_snt:end_snt,0:N_fea],volatile=True).contiguous().cuda(device=device)
                dev_lab= Variable(dev_data_set[dev_beg_snt:end_snt,N_fea],volatile=True).contiguous().long().cuda(device=device)
                if rnn==1:
                  inp=inp.view(inp.shape[0],1,inp.shape[1])
                  lab=lab.view(lab.shape[0],1)
                dev_beg_snt=dev_data_end_index[j]

                [dev_loss,dev_err,dev_pout] = net(dev_inp,dev_lab,test_flag,rank)

                dev_loss_sum=dev_loss_sum+dev_loss.data
                dev_err_sum=dev_err_sum+dev_err.data
                         
                dev_beg_batch=dev_end_batch
             
                dev_end_batch=dev_beg_batch+dev_batch_size
                
          end_val_time,_  = resource_usage(RUSAGE_SELF), timestamp()
          val_time = end_val_time.ru_utime - st_val_time.ru_utime
          sum_val_time=sum_val_time+val_time
          print('[INFO] EPOCH: %d, In Worker: %d, val_Err: %0.3f, val_loss: %0.3f, val_time: %0.3f' % ((e+1), int(rank),dev_err_sum/dev_N_batches, dev_loss_sum/dev_N_batches, sum_val_time))
          dev_err_sum_tot=dev_err_sum/dev_N_batches   
          if e>1:
              threshold = (temp_err-dev_err_sum_tot)/dev_err_sum_tot

              if threshold<0.0005:
                lr = lr * 0.5
          
          net.train()

          beg_batch=0
          end_batch=beg_batch+batch_size
          
          beg_snt=0

          _, st_shu_time= timestamp(), resource_usage(RUSAGE_SELF)
          
          np.random.shuffle(data_set_ori)
          
          if not(save_gpumem):
             data_set=torch.from_numpy(data_set_ori).float().cuda(device=device)
          else:
             data_set=torch.from_numpy(data_set_ori).float()  

          N_fea=data_set.shape[1]-1
          options.input_dim=N_fea
          N_out=int(data_set[:,N_fea].max()-data_set[:,N_fea].min()+1) 
          options.num_classes=N_out
          end_shu_time,_  = resource_usage(RUSAGE_SELF), timestamp()
          shu_time = end_shu_time.ru_utime - st_shu_time.ru_utime
          sum_shu_time=sum_shu_time+shu_time
          loss_sum=0
          err_sum=0

        for i in range(N_batches):

           _, st_load_time= timestamp(), resource_usage(RUSAGE_SELF)

           end_load_time,_  = resource_usage(RUSAGE_SELF), timestamp()
           load_time = end_load_time.ru_utime - st_load_time.ru_utime
           if do_training :
            
            if rnn==1:
             max_len=data_end_index[snt_index+batch_size-1]-data_end_index[snt_index+batch_size-2]
           
             inp= Variable(torch.zeros(max_len,batch_size,N_fea)).contiguous()
             lab= Variable(torch.zeros(max_len,batch_size)).contiguous().long()
           
           
             for k in range(batch_size):
              snt_len=data_end_index[snt_index]-beg_snt
              N_zeros=max_len-snt_len

              N_zeros_left=random.randint(0,N_zeros)

              inp[N_zeros_left:N_zeros_left+snt_len,k,:]=data_set[beg_snt:beg_snt+snt_len,0:N_fea] 
              lab[N_zeros_left:N_zeros_left+snt_len,k]=data_set[beg_snt:beg_snt+snt_len,-1]
              
              beg_snt=data_end_index[snt_index]
              snt_index=snt_index+1
           
           
            else:

             inp= Variable(data_set[beg_batch:end_batch,0:N_fea]).contiguous().cuda(device=device)
             lab= Variable(data_set[beg_batch:end_batch,N_fea]).contiguous().long().cuda(device=device)
            
            
           if do_eval:
              end_snt=data_end_index[i]
              inp= Variable(data_set[beg_snt:end_snt,0:N_fea],volatile=True).contiguous().cuda(device=device)
              lab= Variable(data_set[beg_snt:end_snt,N_fea],volatile=True).contiguous().long().cuda(device=device)
              if rnn==1:
                inp=inp.view(inp.shape[0],1,inp.shape[1])
                lab=lab.view(lab.shape[0],1)
              beg_snt=data_end_index[i]
              

           [loss,err,pout] = net(inp,lab,test_flag,rank)

           if multi_gpu:
             loss=loss.mean()
             err=err.mean()
            
           if do_forward:
            if rnn==1:
               pout=pout.view(pout.shape[0]*pout.shape[1],pout.shape[2]) 
            if int(rank)==1:
              kaldi_io.write_mat(post_file, pout.data.cpu().numpy()-np.log(counts/np.sum(counts)), data_name[i])
            
           if do_training:

            optimizer_worker.zero_grad()  
          

            loss.backward()

            _,st_update_time = timestamp(), resource_usage(RUSAGE_SELF)
            
            main_class.ensure_shared_params(net,rank)
            end_update_time,_  = resource_usage(RUSAGE_SELF), timestamp()
            update_time = end_update_time.ru_utime-st_update_time.ru_utime
            
            
            cc=0
            _,st_model_time = timestamp(), resource_usage(RUSAGE_SELF)

            end_model_time,_  = resource_usage(RUSAGE_SELF), timestamp()
            model_time = end_model_time.ru_utime-st_model_time.ru_utime

            b=0
             

           sum_update_time=sum_update_time + update_time
           sum_load_time=sum_load_time+load_time
           sum_model_time= sum_model_time+model_time
           loss_sum=loss_sum+loss.data
           err_sum=err_sum+err.data

           if i%100==0:
             
             if i!=0:

               print('[INFO] EPOCH: %d, Batch: %d, In Worker: %d, Err: %0.3f, loss: %0.3f, update_time: %0.3f, load_time: %0.3f' % ((e+1),i, int(rank),err_sum/i, loss_sum/i,sum_update_time,sum_load_time))           

           beg_batch=end_batch
           end_batch=beg_batch+batch_size

           m=m+1
        end_epoch_time,_  = resource_usage(RUSAGE_SELF), timestamp()
        epoch_time = end_epoch_time.ru_utime - st_epoch_time.ru_utime
        sum_epoch_time= sum_epoch_time+epoch_time

        if do_training:
            checkpoint={'model_par': net.state_dict(),
                    'optimizer_par' : optimizer_worker.state_dict()}
            torch.save(checkpoint,options.out_file)    

      loss_tot=loss_sum/(N_batches)
      err_tot=err_sum/(N_batches)
      end_train_time,_  = resource_usage(RUSAGE_SELF), timestamp() 
      train_time = end_train_time.ru_utime - st_train_time.ru_utime

      if do_training:
        checkpoint={'model_par': net.state_dict(),
                    'optimizer_par' : optimizer_worker.state_dict()}
        torch.save(checkpoint,options.out_file)

      info_file=out_file.replace(".pkl",".info")

      with open(info_file, "a") as inf:
           inf.write("model_in=%s\n" %(pt_file))
           inf.write("fea_in=%s\n" %(fea_scp))
           inf.write("loss=%f\n" %(loss_tot))
           inf.write("err=%f\n" %(err_tot))
           inf.write("all_time=%f\n" %(train_time))
           inf.write("shu_time=%f\n" %(sum_shu_time))
           inf.write("model load time=%f\n" %(sum_load_time))
           inf.write("gradient send time=%f\n" %(sum_update_time))
           inf.write("val data calculate time=%f\n" %(sum_val_time))
           inf.write("data generate time=%f\n" %(data_time))
           inf.write("model update time=%f\n" %(sum_model_time))
           inf.write("epoch time=%f\n" %((sum_epoch_time-sum_load_time-sum_update_time-sum_model_time-sum_val_time)/num_epoch))
           inf.write("training time=%f\n" %(train_time-sum_load_time-sum_update_time-sum_val_time-data_time-sum_model_time-sum_shu_time))
           
      inf.close()
      
      if do_forward:
          post_file.close()
      

  
  
  
    
def init_processes(rank, world_size, fn, backend,port,ip_add):

      if int(rank)==0:
        print("wait worker")
      options=read_conf()
      do_forward=bool(int(options.do_forward))
      if do_forward and int(rank)==0:
        print("forward")
        fn(rank) 
      elif do_forward and int(rank)!=0:
        print("worker %d end!" % int(rank))  
      
      elif int(world_size)==1:
        print("[INFO] process GPU1")
        fn(rank)
      else:
        print("worker enter: %d" % int(rank))
        distributed.init_process_group(backend=str(backend),init_method='{}://{}:{}'.format( str(backend), str(ip_add), str(port)), rank=int(rank),world_size=int(world_size))
        print("connected worker: ",int(rank))
        fn(rank)

if __name__ == '__main__':
   
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    
    options=read_conf()
    do_forward=bool(int(options.do_forward))
 
    
    port = sys.argv[1]
    world_rank = sys.argv[2]
    world_size = sys.argv[3]
    ip_add = sys.argv[4]
    if do_forward:
      backend='tcp'
      
    else:
      backend='mpi'
      world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
      print(world_size)
      world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    processes = []
    
    Funcc = MAIN_CLASS()
    
    init_processes(world_rank, world_size, Funcc.main, backend,port,ip_add)


    

 

 


  
  
  
  
  
 
