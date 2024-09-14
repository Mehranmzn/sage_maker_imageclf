
# 📊 SageMaker Image Classification

Welcome to the **SageMaker Image Classification** project! This project leverages AWS SageMaker to train and deploy an image classification model using a dataset of chest X-ray images. The model aims to differentiate between normal and pneumonia cases.

## 🚀 Project Overview

This repository contains a Jupyter Notebook that demonstrates how to build an image classification model using Amazon SageMaker, a powerful cloud-based machine learning platform.

### Key Features:
- **AWS SageMaker**: We utilize SageMaker to create and deploy the machine learning model in a scalable manner.
- **Image Processing**: The dataset consists of chest X-ray images that are processed and resized using the `PIL` library for efficient training.
- **Binary Classification**: The model classifies images into two categories: `Normal` and `Pneumonia`.
- **Data Sources**: The dataset is split into training, validation, and test sets, and the results are saved and visualized in SageMaker.

## 📂 Folder Structure

```
├── data/
│   ├── chest_xray/
│   │   ├── train/   # Training dataset (Normal, Pneumonia)
│   │   ├── test/    # Testing dataset (Normal, Pneumonia)
│   │   └── val/     # Validation dataset (Normal, Pneumonia)
├── sage_maker_imageclass.ipynb  # Jupyter Notebook for SageMaker image classification
└── README.md  # This file
```

## 🔧 Setup Instructions

### Prerequisites
To run this project, you'll need the following:

- **AWS Account**: Access to an AWS account with SageMaker enabled.
- **IAM Permissions**: Sufficient IAM permissions to create and run SageMaker jobs.
- **AWS CLI**: Installed and configured with your AWS credentials.
- **Jupyter Notebook**: The notebook `sage_maker_imageclass.ipynb` runs in any Jupyter environment, such as Amazon SageMaker Notebooks, or locally.

### Install Dependencies

Make sure you have the required dependencies installed:

```bash
pip install boto3
pip install sagemaker
pip install matplotlib
pip install pillow
```

### Running the Notebook

1. **Launch Jupyter Notebook**: You can run this notebook on AWS SageMaker, or locally in Jupyter.
2. **Dataset Setup**: Ensure that the dataset is placed in the `data/chest_xray/` folder. If you are using AWS S3, upload the dataset to an S3 bucket and update the notebook to point to your S3 path.
3. **Model Training**: The notebook contains all the steps to preprocess the images, train the model, and evaluate its performance.
4. **Deploying the Model**: Once trained, the model can be deployed on SageMaker for real-time predictions.

## 💡 How It Works

### Data Preprocessing
- **Image Resizing**: The X-ray images are resized to 224x224 pixels to match the input size of the model.
- **Labels**: Each image is labeled as either "Pneumonia" (1) or "Normal" (0) based on the file name.

### Model Training
- The notebook leverages a **pre-trained convolutional neural network (CNN)** from a framework like TensorFlow or PyTorch (depending on the choice in the notebook).
- Fine-tuning is performed using the chest X-ray dataset, with SageMaker handling the heavy lifting in the cloud.

### Evaluation
- The trained model is evaluated on the test dataset to check its accuracy, precision, recall, and other key metrics.
- Predictions are visualized and compared with ground truth labels.

## 🌐 AWS SageMaker Features Used

- **SageMaker Estimators**: To create, train, and tune the machine learning models efficiently.
- **Hyperparameter Tuning**: Optimize model performance by finding the best hyperparameters using SageMaker's built-in capabilities.
- **Model Deployment**: Deploy the trained model to a SageMaker endpoint for real-time predictions.

## 🧠 Dataset

The dataset used for this project contains chest X-ray images categorized as:

- **Normal**: X-rays that do not show signs of pneumonia.
- **Pneumonia**: X-rays showing symptoms of pneumonia.

The dataset can be sourced from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) or other open repositories. Ensure the images are organized into training, validation, and test sets for proper model evaluation.

## 🎨 Visualization

The notebook includes visualization tools that:
- Display random samples of the images.
- Plot accuracy and loss curves.
- Generate confusion matrices to evaluate the model’s performance.

## 📈 Results

The final model's performance is reported with metrics like:
- **Accuracy**: The percentage of correctly classified images.
- **Precision & Recall**: To measure the model's ability to detect pneumonia cases.
- **Confusion Matrix**: A visualization of the model’s predictions vs actual outcomes.

## 🙌 Contribution

Feel free to contribute to this project! Fork the repo, make changes, and submit a pull request. We welcome improvements to the model, additional features, or better visualizations.

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify it as needed.

