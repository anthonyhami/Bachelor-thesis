#! /bin/sh
#BSUB -e /home/fernanda/errors/job_%J.err
#BSUB -o /home/fernanda/errors/job_%J.out
#BSUB -q graphical
#BSUB -R "rusage[mem=80000]"
#BSUB -M 80000
#BSUB -W 30:00
#BSUB -n 10
#BSUB -J AC_GAN_GPU
python PCA_HP_new.py 
echo "Done with submission script"
