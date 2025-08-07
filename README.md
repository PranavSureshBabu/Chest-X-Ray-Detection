# Chest-X-Ray Detection Using Deep Learning

# OVERVIEW

The solution uses deep learning to classify chest X-ray images into Pneumonia and normal categories. Leveraging Convolutional Neural Network, the goal is to assist healthcare professionals in rapid, accurate identification of pneumonia from X-ray scans - potentially reducing diagnostic delays and improving patient outcomes 

<img width="298" height="169" alt="image" src="https://github.com/user-attachments/assets/5daf5b06-49de-4a24-b4ce-1d1aedab2e56" />

# 📂 DATASET

Dataset Source - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

- The dataset is organised into 3 main folders: Train, Test and Val.

- Each of these folders contains two subfolders: NORMAL and PNEUMONIA

- Inside these subfolders are grayscale chest X-ray images representing:

   - NORMAL – patients with no signs of pneumonia.

   - PNEUMONIA – patients diagnosed with bacterial or viral pneumonia.   


# OBJECTIVE

To develop a computer vision based diagonstic support system that can 

- Automatically detect Pneumonia from Chest- X rays

- Reduce the Burden on radiologist by acting as first pass screening tool

- Be easily deployed and accessible through a simple web interface


# IMPLEMENTATION

1. Loading and Preparing the data

To begin, the dataset was carefully loaded using TensorFlow's ImageDataGenerator, which not only reads the images but also rescales pixel values to fall between 0 and 1 — making them ready for training. The   dataset came pre-divided into three folders: train, test, and val, with each folder containing two subfolders for NORMAL and PNEUMONIA cases. This folder structure allowed us to directly load the data while keeping labels intact. All images were resized to a standard 224x224 resolution so that they could be uniformly fed into the neural network.

Why CNN used ?

The model is good at looking images and spotting pattern and also helps in finding key signs of pneumonia(like cloudiness or inflammation) without needing to manually program what to look for 

2.Designing the CNN Architecture

Once the data was ready, a Convolutional Neural Network (CNN) was constructed from scratch. The model started with a series of convolutional layers that automatically learned spatial features from the X-ray images — like edges, textures, and patterns typical of pneumonia. These were followed by max pooling layers that helped reduce the image size and computation, while keeping the important features intact. Dropout layers were added to prevent overfitting, and batch normalization was used to stabilize and speed up training. Finally, a dense layer with a sigmoid activation function was added to output a binary classification: pneumonia or normal.

3. Training the Neural Network

The model was compiled with the Adam optimizer and binary cross-entropy as the loss function — ideal for binary classification tasks. Training was performed over multiple epochs, where the model gradually improved its accuracy by adjusting its internal parameters. During each epoch, it evaluated its performance not just on the training data but also on the validation set to avoid memorizing the data (overfitting). Training and validation accuracy and loss were tracked at every step.

4. Evaluating the Model

Once trained, the model was evaluated on the separate test dataset — a set of X-ray images it had never seen before. This ensured that our results reflected real-world performance, not just memorized patterns. We measured key metrics like accuracy, precision, recall, and also used a confusion matrix to see how well the model performed in distinguishing normal cases from pneumonia.

5. Visualizing Model Insights

To better understand the training process and the model’s behavior, graphs of accuracy and loss over epochs were plotted. These visualizations showed whether the model was learning steadily or overfitting. Additionally, predictions were made on sample test images to visually confirm whether the model could correctly identify them as normal or pneumonia.


# 🛠️ TOOLS & TECHNOLOGIES USED

- Python

- TensorFlow / Keras

- OpenCV & PIL for image handling

- NumPy / Pandas / Matplotlib for data analysis & plotting

- Flask for model deployment as a web application


# RESULTS

- The trained model achieved over 90% accuracy and high recall, meaning it correctly identified pneumonia in the majority of actual cases — minimizing false negatives, which is critical in medical settings where missing a diagnosis can be dangerous.

- By deploying it through a web interface, healthcare workers in rural or underserved areas can upload X-rays and receive instant feedback, even without a radiologist on site — solving a critical access problem in global healthcare.

- The model doesn't replace doctors but provides a "second opinion" tool that can support faster, more consistent diagnoses, especially when human resources are limited or under pressure.


# 🚀 DEPLOYMENT USING FLASK

## 1.Model Saving

- Trained model saved as .h5 file using model.save().

## 2.Flask Application

- Built an interface to upload X-ray images via web UI.

- Flask backend loads the saved model and processes the uploaded image.

- Returns a prediction: Normal or Pneumonia.

- Frontend Integration

- Simple HTML form for uploading images.

- Displays prediction result after model inference.

## 4.Running the App

<img width="753" height="148" alt="image" src="https://github.com/user-attachments/assets/b344a46b-9678-4203-b9a7-b0ccb8301675" />
Navigate to http://127.0.0.1:5000 to use the interface.

<img width="866" height="642" alt="image" src="https://github.com/user-attachments/assets/d8c8beea-d882-4465-b471-ad8726e13fa6" />


# 🔭 Future Scope & Improvements

- Integrate Grad-CAM for visual explanation of predictions.

- Improve generalization using data augmentation and ensemble models.

- Deploy via Docker and host on cloud platforms (AWS/GCP).

- Expand to multi-class detection (e.g., differentiate between bacterial and viral pneumonia).

- Add multilingual support in the UI for global accessibility.

## ✒️ Summary

- Uses deep learning to detect pneumonia from chest X-rays.

- Addresses real-world challenges in early and accurate diagnosis.

- Deployed via Flask for practical, easy-to-use web accessibility.

- Supports healthcare workflows, especially in under-resourced regions.

## CHALLENGES FACED

- One major challenge was loading the dataset directly into Google Colab, as large folder structures can't be uploaded through the Colab interface.

- To overcome this, the entire dataset was uploaded to Google Drive, and the Drive was then mounted in Colab using:
