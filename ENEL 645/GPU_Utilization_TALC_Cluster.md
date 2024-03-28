
# How to Utilize GPU on TALC Cluster 🚀

## 1. Activate Conda 🐍
```
source ~/software/init-conda
```

## 2. Create New Conda Environment 🌱
```
conda create -n <your env name> python=3.11.8 ipython
```

## 3. Activate Newly Created Conda Environment 🔁
Activate the newly created Conda environment to use it for the installation of packages.
```
conda activate <your env name>
```

## 4. Install Conda Packages 📦
Install the main packages needed for deep learning with PyTorch, along with some essential libraries for data processing and visualization.
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install ipykernel wandb Pillow numpy scikit-image scikit-learn matplotlib pytorch-lightning
```
