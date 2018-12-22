<h1>ASGD TIMIT</h1>

- This project is an implementation of the **ASGD** algorithm based on [2].  It used **TIMIT** database to measure the performance of parallel algorithm in the speech recognition task.
- Use **Kaldi** and **PyTorch** open-source library to implement the **triphone DNN-HMM** system.
- Distributed environment is implemented by **PyTorch** multiprocessing and distributed function.
- **run_nn_MPI.py** has modified **mravanelli's pytorch / kaldi / run_nn.py** [1].
- **ps_server.py** is parameter server.
- **run_exp_MPI.sh** is an executable shell script. 
<pre><code>
./run_exp_MPI.sh 'configure file' 'port' 'rank' 'world size' 'ip address'
</code></pre>
- For the remainder of the code, just use the code in [1].



<h1>Reference</h1>

[1] https://github.com/mravanelli/pytorch-kaldi

[2] Zhang, S.; Zhang, C.; You, Z.; Zheng, R.; and Xu, B. 2013. Asynchronous stochastic gradient descent for DNN training. In *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 6660??663.