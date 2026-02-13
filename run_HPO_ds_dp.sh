#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
#SBATCH --time=24:00:00
#SBATCH --job-name=python_ollama_job
#SBATCH --output=output_%j.txt


# Load required modules
module load python/3.12.5
module load ollama/0.11.11

# Start Ollama server in the background
nohup ollama serve > ollama.log 2>&1 &

# Wait a few seconds to ensure Ollama server is running
sleep 5

# Run your Python script
python3 direct_prompting.py --llm deepseek-r1:8b

