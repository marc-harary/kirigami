#!/usr/bin/bash

# Note that the roundaboutness of this script is simply to enable the `$date`
# macro in the log file, which won't be expanded by the `SBATCH` macro

echo '#!/usr/bin/bash' > /tmp/temp.slurm
echo 'python run.py' >> /tmp/temp.slurm
chmod +x /tmp/temp.slurm

if [[ $1 ]]; then
    name="$1"
else
    name="$(date +%F-%T)"
fi

sbatch -p scavenge_gpu \
       --time=2-00:00:00 \
       --mem-per-cpu=64G \
       --gpus=2 \
       --out="%x.log" \
       -J $name \
      /tmp/temp.slurm 
