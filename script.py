import os
os.system('srun -c 2 --gres=gpu:1 -o slurm-test.out -J my_job main.py')