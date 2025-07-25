# Medical-Image-Classification-for-Disease-Detection-CNN-
Medical Image Classification for Disease Detection

🩺 Problem Statement: Disease Detection from Medical Images Using Deep Learning
The objective is to develop a deep learning model to classify medical images (e.g., chest X-rays) to detect diseases such as pneumonia automatically and accurately.
Early and reliable disease detection from medical images can assist healthcare professionals in diagnosis and treatment planning, especially in resource-constrained settings.
The task involves:
Loading and preprocessing medical images by resizing and normalizing pixel values for model readiness.
Splitting the dataset into training, validation, and test sets to evaluate the model reliably.
Building and training a Convolutional Neural Network (CNN) using frameworks like TensorFlow or PyTorch for feature extraction and classification.
Applying data augmentation techniques (flipping, rotation, zoom) to improve generalization and prevent overfitting.
Evaluating the model using metrics such as accuracy, precision, and recall, and visualizing misclassifications to understand the model’s failure points.

🖥️ System Requirements
✅ IDE:
Google Colab with T4 GPU for accelerated deep learning training.
✅ Libraries and Modules Used:
* NumPy & Pandas
  For data loading, manipulation, and preprocessing.
* Seaborn & Matplotlib
  For data visualization, exploratory analysis, and plotting model performance.
* KaggleHub
  For importing and managing datasets efficiently.  
* PIL (Python Imaging Library)
  For image loading, resizing, and visualization.
* TensorFlow
  For building and training Convolutional Neural Networks (CNNs) for medical image classification.
* Scikit-Learn
  For model evaluation using metrics like accuracy, precision, recall, and confusion matrix

🚀 Workflow
1️⃣ Dataset Loading
Imported Chest X-ray dataset using KaggleHub directly into the Colab environment.
2️⃣ Dataset Structure
Dataset organized into train, test, val directories, each further sub-categorized as Normal and Pneumonia.
3️⃣ Data Organization & Visualization
Created separate directories for each class and visualized sample images to understand patterns.
4️⃣ Data Scaling & Augmentation
Applied image scaling and augmentation (rotation, zoom, flips) on training data using ImageDataGenerator to improve generalization.
5️⃣ Preprocessing & Resizing
Resized images to (150, 150) and separated them into training, testing, and validation sets.
6️⃣ Model Building
Built a Sequential CNN using:
Conv2D and MaxPooling2D layers
Hidden layer filters: [32, 64, 128, 256]
Total parameters: ~1,995,649
7️⃣ Model Training
Training setup:
Learning rate: 0.000001
Epochs: 15
Batch size: 32
8️⃣ Model Performance
Achieved 92% accuracy on the testing set.
9️⃣ Evaluation & Analysis
Created a DataFrame with testing data and predicted labels.
Visualized the confusion matrix.
Out of total 79 misclassifications, 5 cases were classified as normal while the patients had pneumonia, indicating a priority focus area for further model tuning.














  






  
