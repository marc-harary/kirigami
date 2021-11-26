#!/usr/bin/bash
#SBATCH -p scavenge_gpu
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=2
#SBATCH --out="%j.out"

cd ~/kirigami.swp/scripts
python train.py ~/kirigami.swp/exps/exp4/args.json
# python thermo_embed.py
# python zuker_embed.py
