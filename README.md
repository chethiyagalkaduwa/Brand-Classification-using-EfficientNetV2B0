# Winter Apparel Brand Classification using EfficientNetV2B0

This repository contains the full implementation for a deep learning pipeline designed to classify winter apparel images by brand using the EfficientNetV2B0 architecture. It leverages transfer learning, custom data augmentation, and class reweighting to train and evaluate a robust multi-class image classifier.

---

Download the dataset folders using the huggingface:
[Chey97/Clothing_Brand_pred](https://huggingface.co/datasets/Chey97/Clothing_Brand_pred)

[![PDF](https://img.shields.io/badge/Report-PDF-blue)](./Model_Report.pdf)


## Project Structure

```bash
.
├── Model_EfficientNetV2B0.ipynb     # Notebook for full model training and experimentation
├── model_testing.py                 # Script for evaluating the trained model on test data
├── Test_user_image.py              # Script to predict brand for a single new image
├── efficientnetv2_finetuned.h5     # Final trained EfficientNetV2B0 model
├── class_indices_effnet.npy        # NumPy array of class name to index mapping
└── README.md                       # Project overview and usage instructions
````

---

## Dataset Overview

* **Training Dataset**: \~2,700 images scraped from eBay, covering 34 brands with \~80 samples per class.
* **Test Dataset**: 887 images provided separately for final evaluation.
* Each image is labeled as: `"filename.jpg" ["brand"]`

---

## Training the Model

To retrain the model, open and run the Jupyter notebook:

```bash
Model_EfficientNetV2B0.ipynb
```

Key features:

* Uses EfficientNetV2B0 pretrained on ImageNet
* Includes custom classification head
* Applies data augmentation and class weights to handle imbalance
* EarlyStopping and ReduceLROnPlateau for optimized training

---

## Testing the Model

To evaluate the model on your test set:

```bash
python model_testing.py
```

Ensure your test directory and metadata file are properly configured within the script.

---

## Predict on a New Image

To run a single prediction on a user-provided image:

```bash
python Test_user_image.py --image path_to_your_image.jpg
```

This script loads the trained model, preprocesses the image, and returns the top brand prediction.

---

## Requirements

* Python 3.8+
* TensorFlow 2.x
* scikit-learn
* NumPy
* Pillow
* Matplotlib
* tqdm

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Model Performance

* **Training Accuracy**: \~77%
* **Validation Accuracy**: \~25%
* **Test Accuracy**: \~13.87%

---

## Notes

* Due to dataset imbalance and visual brand similarity, generalization is limited.
* Future improvements may involve vision transformers, synthetic data, or contrastive learning.

---
