cd /root/notebooks/imet/iMet-Collection-2019-FGVC6

bash /root/notebooks/install_tf2.sh

pip install runipy

pip install torchsummary

pip install pretrainedmodels

pip install numpy==1.17.3

pip install albumentations

pip install prefetch_generator

#python imet_top_solution.py

#runipy runipy.ipynb OutputNotebook.ipynb

#runipy runipy.ipynb

mkdir -p /home/aiforge/.cache/torch/checkpoints/

cp se_resnext101_32x4d-3b2fe3d8.pth /home/aiforge/.cache/torch/checkpoints/

# runipy  -o imet_top_solution3.ipynb
python imet_top_solution4.py