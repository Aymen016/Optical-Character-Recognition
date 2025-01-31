# 🖼️ Optical Character Recognition (OCR) System

## 🌟 About
This project implements a **Naive Bayes-based OCR System** that recognizes handwritten digits (2 and 4). The goal is to process images, extract features, and predict digits based on pixel values. It’s a simple yet effective application of machine learning techniques to solve Optical Character Recognition tasks.

## 🚀 How It Works
The OCR system processes images of handwritten digits, extracts pixel data, and applies the **Naive Bayes classifier** to predict the digit in the image. By analyzing the probability of pixels being part of specific digits, the model classifies the image based on the highest likelihood.

### Key Features:
- **Image Preprocessing**: Converts raw pixel data into feature vectors suitable for classification.
- **Naive Bayes Classifier**: A powerful algorithm that leverages Bayes' Theorem to classify images based on pixel likelihoods.
- **Data Handling**: Handles both training and test data by splitting based on classes (2 and 4), and calculates class and feature probabilities.

## 📊 Model Evaluation
After training, the model's performance is evaluated based on:
- **Confusion Matrix**: Shows true positive (TP), false positive (FP), true negative (TN), and false negative (FN) values.
- **Accuracy**: Measures the proportion of correct predictions made by the model on the test dataset.
- **Classwise Accuracy**: Evaluates how well the model predicts each digit class (2 and 4).

## 💡 Why Naive Bayes Works Well for OCR
Naive Bayes is particularly suited for OCR tasks due to its ability to handle:
- **Feature Independence**: Each pixel is considered independently, making it computationally efficient for high-dimensional data.
- **Small Training Data**: Even with limited training data, Naive Bayes performs effectively, which is useful for OCR where labeled data might be sparse.

### Limitations of Naive Bayes in OCR:
While Naive Bayes excels in simple cases, it may struggle with more complex features or where the relationships between pixels are not independent (e.g., continuous data such as house prices).

## 🚧 Future Improvements
- **Advanced Techniques**: Experiment with **deep learning models** such as **Convolutional Neural Networks (CNNs)** for more accurate and scalable OCR solutions.
- **Hyperparameter Tuning**: Enhance the model’s performance by tuning hyperparameters and experimenting with different machine learning algorithms.
- **Real-Time OCR**: Develop a **real-time OCR system** for applications like document scanning and live text recognition.

## 🎯 Conclusion
This OCR system serves as an excellent starting point for recognizing handwritten digits using a **Naive Bayes classifier**. It demonstrates the efficiency of Naive Bayes in solving image classification tasks and lays the foundation for future work in more advanced OCR systems.

Feel free to **fork** this repository, **contribute**, and make enhancements! Let's build better OCR solutions together. 🤝✨

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
