---

# Garbage Classification 🦸‍♂️🌍

## Objective 🎯

Develop a PyTorch model to classify images of garbage. By participating in this eco-friendly initiative, we're not just coding; we're making the world a cleaner place!

## Requirements 📋

- PyTorch
- Jupyter Notebook

## Team Members 👩‍💻👨‍💻

- Alton Wong
- Carissa Chung
- Christian Valdez
- Redge Santillan

(Replace with your actual team members' names or nicknames!)

## Setup 🛠️

### Jupyter Setup

1. **Install Jupyter Notebook**
2. **Clone Our Repository**: `git clone https://github.com/chvaldez10/ENSF-611-ENEL-645.git`
3. **Navigate to Our Project**: `cd ENEL\ 645/A2/`
4. **Launch Jupyter Notebook**: `jupyter notebook`

### Python Project Setup

Ensure you have at least Python 3.10.0 installed.

Optionally, you can set up a virtual environment using conda or pyenv. For this project, we're using conda:

```bash
conda create --name enel_645 python=3.11.7
conda activate enel_645
pip install -r requirements.txt
```

### TALC

We'll be outsourcing our heavy-duty tasks to TALC. If you're not connected to the UofC network, download [FortiClient VPN](https://ucalgary.service-now.com/it?id=kb_article&sys_id=52a169d6dbe5bc506ad32637059619cd). SSH into the cluster:

```bash
ssh user.name@talc.ucalgary.ca
```

### Managing Configurations

To manage our configurations, we'll be using a `.env` file. Create a `.env` file with the following information:

```bash
DATASET_LOCAL_PATH=
DATASET_REMOTE_PATH=
TALC_USER_LOGIN=
```
