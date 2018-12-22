#!/bin/bash


           
function write_conf {
 
    echo "[todo]" > $conf_file
    echo "do_training=$do_training" >> $conf_file
    echo "do_eval=$do_eval" >> $conf_file
    echo "do_forward=$do_forward" >> $conf_file
    echo " " >> $conf_file
    
    echo "[data]" >> $conf_file
    echo "fea_scp=$fea_chunk" >> $conf_file
    echo "fea_opts=$fea_opts" >> $conf_file
    echo "lab_folder=$lab_folder" >> $conf_file
    echo "lab_opts=$lab_opts" >> $conf_file
    echo "dev_fea_scp=$dev_fea_chunk" >> $conf_file
    echo "dev_fea_opts=$dev_fea_opts" >> $conf_file
    echo "dev_lab_folder=$dev_lab_folder" >> $conf_file
    echo "dev_lab_opts=$dev_lab_opts" >> $conf_file
    echo "pt_file=$pt_file" >> $conf_file
    echo "count_file=$count_file" >> $conf_file
    echo "out_file=$out_file" >> $conf_file
    echo " " >> $conf_file
    
    echo "[architecture]" >> $conf_file
    echo "NN_type=$NN_type" >> $conf_file
    echo "cnn_pre=$cnn_pre" >> $conf_file
    echo "hidden_dim=$hidden_dim" >> $conf_file
    echo "N_hid=$N_hid" >> $conf_file
    echo "drop_rate=$drop_rate" >> $conf_file
    echo "use_batchnorm=$use_batchnorm" >> $conf_file
    echo "use_laynorm=$use_laynorm" >> $conf_file
    echo "cw_left=$cw_left" >> $conf_file
    echo "cw_right=$cw_right" >> $conf_file
    echo "seed=$seed" >> $conf_file
    echo "use_cuda=$use_cuda" >> $conf_file
    echo "bidir=$bidir" >> $conf_file
    echo "resnet=$resnet" >> $conf_file
    echo "skip_conn=$skip_conn" >> $conf_file
    echo "act=$act" >> $conf_file
    echo "act_gate=$act_gate" >> $conf_file
    echo "resgate=$resgate" >> $conf_file
    echo "minimal_gru=$minimal_gru" >> $conf_file
    echo "cost=$cost" >> $conf_file
    echo "twin_reg=$twin_reg" >> $conf_file
    echo "twin_w=$twin_w" >> $conf_file
    echo "multi_gpu=$multi_gpu" >> $conf_file
    echo " " >> $conf_file
    
    echo "[optimization]" >> $conf_file
    echo "lr=$lr" >> $conf_file
    echo "optimizer=$optimizer" >> $conf_file
    echo "batch_size=$batch_size" >> $conf_file
    echo "save_gpumem=$save_gpumem" >> $conf_file
}
           
           
# Reading Param File
cfg_file=$1
port=$2
rank=$3
world_size=$4
ip_add=$5
cmd=""

# Parsing cfg file
source <(grep = $cfg_file)
IFS=, read -a tr_fea_scp_list <<< $tr_fea_scp
IFS=, read -a dev_fea_scp_list <<< $dev_fea_scp
IFS=, read -a te_fea_scp_list <<< $te_fea_scp

# creating output folder
mkdir -p $out_folder

# Initialization
pt_file='none'

# Number of training chunks
N_ck=${#tr_fea_scp_list[@]}

sleep_time=3

echo 'Training...'

for epoch in $(seq -w 1 $N_ep); do 
  echo "[INFO] epoch: " + $epoch
  if [ "$epoch" -gt "1" ]; then
    err_dev_prev=$err_dev
  fi

  for chunk in $(seq -w 0 "$(($N_ck-1))"); do 
    
    fea_chunk=${tr_fea_scp_list[$chunk]}
    fea_opts=$tr_fea_opts
    lab_folder=$tr_lab_folder
    lab_opts=$tr_lab_opts
    
    
    
    out_file=$out_folder"/train_ep_"$epoch"_ck_"$chunk".pkl"
    info_file=$out_folder"/train_ep_"$epoch"_ck_"$chunk".info"
    conf_file=$out_folder"/train_ep_"$epoch"_ck_"$chunk".cfg"

  
  
    do_training=1
    do_eval=0
    do_forward=0
    
  
    write_conf


    $cmd mpirun -np 5 python3 run_nn_MPI.py $port $rank $world_size $ip_add --cfg $conf_file 2> $out_folder/log.log || exit 1

    
    while [ ! -f $info_file ]
    do
     sleep $sleep_time
    done


 
    seed=$(($seed+100))
    
 
    if [ "$epoch" -gt "0" ]; then
   
     echo "[INFO]epoch: " $epoch
    fi
    
    pt_file=$out_file

  done
  
  # Computing total training loss
  loss_tr="$(cat $out_folder"/train_ep_"$epoch"_ck_"*".info" | grep 'loss=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum / NR }')"
  err_tr="$(cat $out_folder"/train_ep_"$epoch"_ck_"*".info" | grep 'err=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum / NR }')"
  time_tr="$(cat $out_folder"/train_ep_"$epoch"_ck_"*".info" | grep 'time=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum}')"

  # eval config
  do_training=0
  do_eval=1
  do_forward=0


  write_conf
  sleep 2

    # Computing total dev loss
  echo "[INFO]Computing total dev loss"

  do_training=0
  do_eval=1
  do_forward=0

  
  # writing config file for eval
  fea_chunk=$te_fea_scp
  fea_opts=$te_fea_opts
  lab_folder=$te_lab_folder
  lab_opts=$te_lab_opts
    
  conf_file=$out_folder"/test_ep_"$epoch"_ck_"$chunk".cfg"

  write_conf

  # Computing total test loss
  loss_te="$(cat $out_folder"/test_ep_"$epoch"_ck_"*".info" | grep 'loss=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum / NR }')"
  err_te="$(cat $out_folder"/test_ep_"$epoch"_ck_"*".info" | grep 'err=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum / NR }')"
  time_te="$(cat $out_folder"/test_ep_"$epoch"_ck_"*".info" | grep 'time=' | awk -F "=" '{ sum += $2 } END { if (NR > 0) print sum}')"
  
  
  printf "epoch %s tr_loss=%s tr_err=%s dev_err=%s test_err=%s learning_rate=%s time=%s sec. \n" $epoch $loss_tr $err_tr $err_dev $err_te $lr $time_tr >>$out_folder/res.res
  
  # Learning Rate Annealing
  if [ "$epoch" -gt "1" ]; then
   relative_imp=`echo "(($err_dev_prev-$err_dev)/$err_dev)<$improvement_threshold" | bc -l`
  if [ "$relative_imp" -eq "1" ]; then
   lr=`echo "$lr*$halving_factor" | bc -l`
  fi
 fi
done

if [ $3 -eq 0 ]; then
echo 'Forward...'

do_forward=1
conf_file=$out_folder"/forward_ep_"$epoch"_ck_"$chunk".cfg"
out_file=$out_folder"/forward_ep_"$epoch"_ck_"$chunk".pkl"
info_file=$out_folder"/forward_ep_"$epoch"_ck_"$chunk".info"

#[ -e $conf_file ] && rm $conf_file

write_conf


$cmd python3 run_nn_MPI.py $port $rank $world_size $ip_add --cfg $conf_file 2> $out_folder/log.log || exit 1

while [ ! -f $info_file ]
   do
     sleep $sleep_time
done


  echo 'Decoding..'
  # Decoding
  cd kaldi_decoding_scripts
  $cmd ./decode_dnn_TIMIT.sh $graph_dir $data_dir $ali_dir $out_folder/decoding_test "cat $out_folder"/forward_ep_"$epoch"_ck_"$chunk".pkl""
  wait
fi

touch MLP_mfcc_128_h4_e24_GPU4_1024
cat $out_folder/decoding_test/score_*/ctm_39phn.filt.sys | grep Sum/Avg > MLP_mfcc_128_h4_e24_GPU4_1024
cat MLP_mfcc_128_h4_e24_GPU4_1024
