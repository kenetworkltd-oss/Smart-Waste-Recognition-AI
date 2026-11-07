# Smart Waste Recognition AI

## 1. Project Overview

Smart Waste Recognition AI is a computer vision and deep learning project developed to classify waste materials into nine categories using Convolutional Neural Networks (CNNs).

The project uses the UCI RealWaste dataset and PyTorch to develop, train, evaluate and improve image classification models. OpenCV is also integrated to support real-time image capture and classification through a webcam.

The main objective is to explore how deep learning and computer vision can support automated waste identification and contribute to smarter waste management systems.

## 2. Project Objectives

- Develop an image classification model capable of recognising nine waste categories.
- Understand computer vision and CNN architecture through practical implementation.
- Build and compare multiple CNN models.
- Improve model performance through architectural improvements and fine-tuning.
- Evaluate the final model using classification metrics.
- Implement real-time waste image classification using OpenCV.
- Prepare the project for future deployment and further development.

## 3. Project Development Journey

The project began with the goal of developing an AI model capable of recognising different waste materials from images.

The initial approach considered using a pretrained deep learning model through transfer learning. However, computational limitations made working with the intended pretrained approach impractical in the available development environment.

The development environment relied on Google Colab with CPU-based computation and no NVIDIA GPU.

Rather than allowing this limitation to stop the project, the approach was adjusted to building CNN architectures from scratch using PyTorch.

This provided an opportunity to understand the internal workings of convolutional neural networks, including feature extraction, pooling, activation functions, regularisation, training and evaluation.

The project subsequently progressed through three CNN architectures, an improved version of CNN Model 3 and a final fine-tuned version.

## 4. Dataset Description

The project uses the **UCI RealWaste dataset**, containing 4,752 real-world waste images distributed across nine categories.

The dataset includes:

| Waste Category | Class Index | Images |
|---|---:|---:|
| Cardboard | 0 | 461 |
| Food Organics | 1 | 411 |
| Glass | 2 | 420 |
| Metal | 3 | 790 |
| Miscellaneous Trash | 4 | 495 |
| Paper | 5 | 500 |
| Plastic | 6 | 921 |
| Textile Trash | 7 | 318 |
| Vegetation | 8 | 436 |
| **Total** | | **4,752** |

Dataset source: [UCI Machine Learning Repository – RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste)

## 5. Dataset Inspection and Verification

Before training, the dataset underwent inspection and verification to understand its structure and quality.

The following checks were performed:

- Total number of images and class distribution.
- Image readability and file integrity.
- Duplicate image detection using MD5 hashing.
- Image dimensions and colour channels.
- Verification of image formats.

The inspection confirmed:

- 4,752 images.
- Zero unreadable images.
- Zero detected duplicate images.
- All images were RGB.
- Typical image dimensions were 524 × 524 pixels.

## 6. Data Preparation and Preprocessing

The dataset was organised into features (X) and target labels (y).

It was divided into training, validation and testing sets using stratified splitting to maintain class distribution.

| Dataset | Images |
|---|---:|
| Training | 3,326 |
| Validation | 713 |
| Testing | 713 |
| **Total** | **4,752** |

The random state was set to 42 to ensure reproducibility.

Image preprocessing included:

- Resizing images to 224 × 224 pixels.
- Converting images into PyTorch tensors.
- Normalising image channels using ImageNet mean and standard deviation.
- Applying data augmentation to training images using random horizontal flipping, rotation and colour jitter.

PyTorch DataLoaders were used to organise images into batches for model training and evaluation.

## 7. CNN Model Development

Instead of relying on pretrained architectures, three CNN models were developed from scratch using PyTorch.

Each model represented a stage of architectural experimentation and performance improvement.

### 7.1 CNN Model 1

The first CNN was developed as the baseline architecture.

Its purpose was to establish a starting point for understanding how convolutional layers extract visual features and how a neural network learns to classify waste images.

The model provided an initial performance benchmark for subsequent architectures.

### 7.2 CNN Model 2

The second CNN architecture introduced further improvements to the network design.

The objective was to improve feature extraction and classification performance compared with the baseline model.

The model was trained and evaluated to understand how architectural changes affected learning and generalisation.

### 7.3 CNN Model 3

The third CNN introduced a more developed architecture with additional convolutional layers.

Its architecture included:

- Convolutional layers.
- Batch Normalisation.
- ReLU activation.
- Max Pooling.
- Adaptive Average Pooling.
- Dropout.
- Fully Connected Classification Layer.

The model was designed to extract increasingly complex image features while using regularisation techniques to help reduce overfitting.

### 7.4 Improved CNN Model 3

Following the evaluation of the original CNN Model 3, an improved version was developed.

The objective was to address performance limitations and improve classification results through further model experimentation and training improvements.

The improved model achieved better test accuracy than the original CNN Model 3.

### 7.5 Fine-Tuning Improved CNN Model 3

The improved CNN Model 3 was subsequently fine-tuned to further optimise its performance.

Fine-tuning involved continuing model optimisation and monitoring validation performance to identify the best-performing checkpoint.

The best checkpoint was selected at fine-tuning Epoch 9.

This version became the final selected model for the project.

## 8. Model Evaluation and Comparison

The models were evaluated using the test dataset to measure their classification performance.

Evaluation included:

- Test loss.
- Accuracy.
- Precision.
- Recall.
- F1-score.
- Classification reports.
- Confusion matrices.
- ROC-AUC.

The evaluation process helped identify differences in performance across the nine waste categories and highlighted areas where the models struggled to classify certain materials.

### 8.1 Model Accuracy Comparison

| Model | Test Accuracy |
|---|---:|
| Original CNN Model 3 | 51.89% |
| Improved CNN Model 3 | 63.25% |
| Fine-Tuned CNN Model 3 | 66.62% |

The fine-tuned CNN Model 3 achieved the highest test accuracy and was selected as the final model.

### 8.2 Final Model Performance

| Evaluation Metric | Result |
|---|---:|
| Test Loss | 0.9950 |
| Test Accuracy | 66.62% |
| Weighted Precision | 67.18% |
| Weighted Recall | 66.62% |
| Weighted F1-score | 65.42% |
| ROC-AUC | 93.19% |

The final model correctly classified 475 out of 713 test images.

The evaluation also revealed differences in class-level performance. Vegetation achieved strong recall, while Miscellaneous Trash remained a challenging category.

These results demonstrate that overall accuracy alone does not fully describe model performance, making class-level evaluation important.

## 9. Final Model Selection and Saving

The fine-tuned CNN Model 3 was selected based on its performance against the other developed versions.

The final model checkpoint was saved for future inference and deployment.

**Saved model:**

`final_cnn_model3.pth`

The checkpoint is stored in the project's `colab_models` directory.

## 10. Real-Time Computer Vision and Waste Classification

OpenCV was integrated to extend the project beyond static dataset evaluation.

Since Google Colab runs in a remote environment, browser-based JavaScript was used as a webcam access bridge, while Python handled image processing and model inference.

The real-time classification pipeline included:

1. Accessing the webcam.
2. Capturing an image.
3. Decoding the captured image.
4. Converting the image from BGR to RGB.
5. Resizing and normalising the image using the existing preprocessing pipeline.
6. Adding a batch dimension.
7. Passing the image through the final CNN model.
8. Generating prediction probabilities.
9. Identifying the predicted waste category and confidence score.
10. Displaying the prediction on the captured image using OpenCV.

Reusable Python functions were developed to simplify repeated image capture, preprocessing and prediction.

This allowed different waste materials to be tested without repeatedly writing the entire prediction pipeline.

**Important distinction:** The current CNN performs image classification. It predicts the dominant waste category in the supplied image; it does not locate individual objects using bounding boxes.

## 11. Technologies and Libraries

The project was developed using the following tools and technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | CNN development, training and inference |
| Torchvision | Image transformations and preprocessing |
| NumPy | Numerical operations |
| Pandas | Data organisation and analysis |
| Matplotlib | Data visualisation |
| Seaborn | Evaluation visualisations |
| Scikit-learn | Dataset splitting and evaluation metrics |
| OpenCV | Image processing and webcam integration |
| PIL (Pillow) | Image handling |
| Google Colab | Model development and training |
| Google Drive | Dataset, notebook and model storage |
| Git and GitHub | Version control and project publication |

## 12. Project Structure

```text
Smart Waste Recognition AI/
│
├── colab_models/
│   └── final_cnn_model3.pth
│
├── deployment/
│   ├── app.py
│   └── requirements.txt
│
├── live_capture.jpg
├── plastic_prediction.jpg
│
├── Smart Waste Recognition AI.ipynb
│
└── README.md
```

*The structure may expand as additional experiments, documentation and deployment files are added.*

## 13. Installation and Usage

The model was developed and tested in Google Colab.

To explore the project:

1. Clone or download the GitHub repository.
2. Open the Jupyter Notebook in Google Colab or a compatible Jupyter environment.
3. Install the required dependencies.
4. Download the RealWaste dataset from the UCI Machine Learning Repository.
5. Update the dataset path to match your environment.
6. Run the notebook sections to inspect the dataset, train the CNN models and evaluate their performance.

The final trained checkpoint can also be used for image classification inference without retraining the model, provided the CNN architecture and preprocessing pipeline are recreated correctly.

## 14. Limitations and Future Improvements

Although the project achieved 66.62% test accuracy, there are opportunities for further improvement.

Potential future developments include:

- Improving classification performance across underperforming categories.
- Experimenting with transfer learning using more suitable computational resources.
- Comparing CNN architectures with Vision Transformers.
- Exploring object detection using YOLO.
- Combining object detection with CNN-based classification.
- Building a user-friendly web application.
- Deploying the final model for public access.
- Exploring integration with automated waste sorting and robotics systems.

## 15. Conclusion

Smart Waste Recognition AI provided practical experience in developing a computer vision solution from dataset inspection through CNN development, training, evaluation and real-time image classification.

The decision to build CNN architectures from scratch, following computational limitations with the initial pretrained-model approach, provided a deeper understanding of neural network development.

Through successive architectural improvements and fine-tuning, the project achieved a final test accuracy of 66.62%.

Beyond the final performance, the project demonstrates a complete practical learning process involving data preparation, deep learning, computer vision, model evaluation and real-time inference.

## 16. Author 

** Agboola Kehinde Oluwaseun [Flyvark I]

**Project:** Smart Waste Recognition AI

**Focus Areas:** Computer Vision, Deep Learning, CNNs, Image Classification and Artificial Intelligence.

**Framework:** PyTorch

**Repository:** [GitHub repository link to be added]

**Project Status:** Core model development and real-time image classification completed. Full web deployment planned for a future portfolio development stage.
