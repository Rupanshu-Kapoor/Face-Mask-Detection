# Face Mask Detection App

## Overview
This Face Mask Detection App is designed to identify whether individuals are wearing face masks in images or real-time video feeds. The app utilizes a custom Convolutional Neural Network (CNN) model trained on an augmented [dataset](https://www.kaggle.com/datasets/rupanshukapoor/face-mask-detection) to achieve high accuracy. It employs HaarCascade classifiers for face detection and Streamlit for an interactive and user-friendly interface. This application is particularly useful in public safety scenarios, such as during the COVID-19 pandemic, to ensure compliance with mask-wearing protocols in public spaces.

## Features
- **Real-time Detection**: Detects faces and verifies mask-wearing in live video streams.
- **Image Detection**: Processes and analyzes static images to determine mask usage.
- **High Accuracy**: Achieved through data augmentation and a custom CNN model.
- **User-Friendly Interface**: Built with Streamlit for easy interaction and deployment.

## Technologies Used
- **Programming Language**: Python
- **Machine Learning Framework**: TensorFlow / Keras
- **Face Detection**: OpenCV (Haar Cascade Classifier)
- **Web Framework**: Streamlit
- **Data Manipulation**: NumPy, Pandas
- **Data Visualization**: Matplotlib, Seaborn
- **Additional ML Utilities**: Scikit-learn
- **Data Augmentation**: ImageDataGenerator (Keras)

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/face-mask-detection-app.git

2. Navigate to the project directory:
   ```
   cd face-mask-detection-app
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
4. Download the [dataset](https://www.kaggle.com/datasets/rupanshukapoor/face-mask-detection) and place in inside `data` folder if you want to train your own model.


## Usage
1. To run the app, use the following command:
   ```
   streamlit run app.py
2. Follow the on-screen instructions to upload an image or start the webcam for real-time detection

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.


## Contact
For any questions or inquiries, please contact:

Name: Rupanshu Kapoor
Email: rupanshukapoor@outlook.com
