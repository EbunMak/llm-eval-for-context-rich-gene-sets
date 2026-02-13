#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --partition=bigmem
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
python3 main_llm.py --llm qwen3:32b

