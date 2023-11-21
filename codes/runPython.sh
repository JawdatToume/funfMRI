#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name="200 Lin Reg Training Sub 1"
#SBATCH --mail-user="ins@ualberta.ca"
#SBATCH --mail-type="all"

module load python
source ~/ENV/bin/activate

cd ~/projects/def-popescuc/smithi23/CMPUT624

python ccRun.py

deactivate
