#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name="slir on convnext features"
#SBATCH --mail-user="sbhuiya@ualberta.ca"
#SBATCH --mail-type="all"

module load python/3.9

cd ~/projects/def-afyshe-ab/sbhuiya/funfmri

source slir_on_fmri_data/bin/activate

python slir_on_convnext_features.py

deactivate
