
# How to Utilize GPU on TALC Cluster 🚀

## 1. Activate Conda 🐍
```
source ~/software/init-conda
```

## 2. Create New Conda Environment 🌱
```
conda create -n <your env name>
```

## 3. Install Conda Packages 📦
Install the main packages needed for deep learning with PyTorch, along with some essential libraries for data processing and visualization.
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
conda install ipykernel wandb Pillow numpy scikit-image scikit-learn matplotlib pytorch-lightning
```
