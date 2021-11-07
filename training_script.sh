#!/bin/bash
#SBATCH -J training
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_1d2g
#SBATCH -c 2
#SBATCH -N 1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

#source /home/senmaoye/.bashrc

#cd jay/multitask

#conda activate torch

nvidia-smi

python train.py --epoch 100 --seed 3 --b 128 --lr 0.0001 --weight_d 0 --gpu 1 --data_path '../1_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt' --save_path 'setting1'
python train.py --epoch 100 --seed 3 --b 128 --lr 0.0001 --weight_d 0 --gpu 1 --data_path '../2_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt' --save_path 'setting2'
python train.py --epoch 100 --seed 3 --b 128 --lr 0.0001 --weight_d 0 --gpu 1 --data_path '../3_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt' --save_path 'setting3'
python train.py --epoch 100 --seed 3 --b 128 --lr 0.0001 --weight_d 0 --gpu 1 --data_path '../4_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt' --save_path 'setting4'
python train.py --epoch 100 --seed 3 --b 128 --lr 0.0001 --weight_d 0 --gpu 1 --data_path '../5_new_separate_normal_myTensor_Log_power_88_bird_acoustic.pt' --save_path 'setting5'
