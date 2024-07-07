conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
conda install -c conda-forge ninja -y
pip install -r requirements.txt
pip install flash-attn
conda install -c conda-forge openmpi -y