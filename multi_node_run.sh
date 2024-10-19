#!/bin/bash 
master_address='128.143.63.115'
n_nodes=4
p=gnolim
node=ai

srun -w ${node}'01' -p $p --gres=gpu:4 --mem=40G -t 4-00:00:00 python run.py -m DP-LDM -dn mnist_28 -e 10.0 setup.node_rank=0 setup.n_nodes=$n_nodes setup.n_gpus_per_node=4 setup.master_address=$master_address &
sleep 20
srun -w ${node}'02' -p $p --gres=gpu:4 --mem=40G -t 4-00:00:00 python run.py -m DP-LDM -dn mnist_28 -e 10.0 setup.node_rank=1 setup.n_nodes=$n_nodes setup.n_gpus_per_node=4 setup.master_address=$master_address &
srun -w ${node}'03' -p $p --gres=gpu:4 --mem=40G -t 4-00:00:00 python run.py -m DP-LDM -dn mnist_28 -e 10.0 setup.node_rank=2 setup.n_nodes=$n_nodes setup.n_gpus_per_node=4 setup.master_address=$master_address &
srun -w ${node}'04' -p $p --gres=gpu:4 --mem=40G -t 4-00:00:00 python run.py -m DP-LDM -dn mnist_28 -e 10.0 setup.node_rank=3 setup.n_nodes=$n_nodes setup.n_gpus_per_node=4 setup.master_address=$master_address &
#srun -w ai04 -p gnolim python main.py --mode train setup.node_rank=2 &
#srun -w ai05 -p gnolim python main.py --mode train setup.node_rank=3 &

wait
# hostname -I 查看addr
# jaguar03 128.143.136.9
# lynx01 128.143.69.60
# ai01 128.143.63.115
# ai04 128.143.63.198
# affogato12 128.143.69.81
# affogato11 128.143.69.75