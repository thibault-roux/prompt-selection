#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --output="output.txt"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:TeslaV100:2
#SBATCH --qos="preemptible"
#SBATCH --mem-per-cpu=10000
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

source ~/.bash_profile
source ../envprompt/bin/activate
OLLAMA_DEBUG=1
ollama serve &
sleep 60
python infer.py