# The results in Table 6 and 7
python run.py setup.n_gpus_per_node=4 --method PDP-Diffusion --dataset_name mnist_28 --epsilon 10.0 

# The results in Table 5
python run.py setup.n_gpus_per_node=4 --method PDP-Diffusion --dataset_name mnist_28 --epsilon 10.0  eval.mode=syn
