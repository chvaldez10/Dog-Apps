# ENEL 645 Final Project: Enhancing Dog Breed Classification 🐕💡

## Team Members 🧑‍🔬

- **Carissa Chung**
- **Redge Santillan**
- **Christian Valdez**
- **Alton Wong**

## Getting Started 🚀

Ensure you have a stable version of Python (3.10+ is recommended).

Then, install all necessary libraries using the command:

```bash
pip install -r requirements.txt
```

We experimented with different models:

- **InceptionV3**
- **ResNet-50**
- **Inception-ResNet-v2**
- **EfficientNet**
- **VGG16**
- **Xception**

The scripts for the models are located inside the `project` folder. Run the models using `<model_name>.py`.

For those using slurm clusters, you can find slurm files under `project/shlurm` to execute the scripts.

## Abstract 📄

This project addresses the challenge of classifying 142 different dog breeds, utilizing a dataset with approximately 100 images per breed. Considering the dataset's limitations, we leverage multiple pre-trained deep learning models to enhance our classification accuracy. Preliminary tests focused on assessing the models' performance in terms of accuracy, loss, and generalizability, with one model, in particular, showing exceptional promise. Further insights are detailed in our comprehensive report.

We utilized the [143 Different Dog Breeds](https://www.kaggle.com/datasets/rafsunahmad/143-different-dog-breeds-image-classifier) dataset found on Kaggle.

## Introduction 📘

Dog breed classification presents a unique challenge in computer vision, characterized by the subtle yet significant variability within and among breeds. This project strives to differentiate these nuances, such as fur color, size, and facial features, by employing advanced deep learning models. This effort not only advances the field of computer vision but also holds significant implications for veterinary medicine, animal welfare, and the pet industry by potentially enhancing the care and understanding of dogs worldwide.

### Background 🌐

Accurately classifying dog breeds is essential due to their high intra-class variability and inter-breed similarities. This project aims to surpass traditional computer vision limitations by employing specialized deep learning models, representing a notable advancement in the precise classification of dog breeds based on nuanced visual cues.

### Problem Statement 🚩

The primary challenge stems from the vast diversity and subtle differences within and across breeds, necessitating the development of sophisticated models for precise classification.

### Objectives 🎯

Our objective is to develop a deep learning model capable of accurately classifying dog breeds from images. We aim for high precision and recall in predicting breed-specific features, comparing our model to existing methods to highlight progress in dog breed classification.

### Significance of the Study 🔍

Enhancing the classification of dog breeds, particularly rare and lesser-known ones, can significantly improve breed recognition systems. This advancement aims to overcome current limitations, contributing to the betterment of dogs' welfare globally.
