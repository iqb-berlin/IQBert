  GNU nano 7.2                                                                                        job.slurm
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --partition=gpu
#SBATCH --account=francphi
#SBATCH --gres=gpu

source ~/myenv/bin/activate
python3 test.py
