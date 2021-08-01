# Dog Breed Classifier
`The capstone project for Udacity’s Machine Learning Engineering Nanodegree Program`

### Table of Contents
1. [Project Overview](#overview)
2. [Problem Statement](#statement)
3. [Data Exploration and Visualization](#eda)
4. [Human Detector](#human)
5. [Dog Detector](#dog)
6. [List of Dependencies](#dependency)
7. [File Descriptions](#desc)
8. [Conclusion](#conc)
9. [Tips to improve the performance](#improve)
10. [Licensing, Authors, and Acknowledgements](#licensing)


## Project Overview<a name="overview"></a>
In this project, I have implemented an `end-to-end deep learning pipeline` that can be used **within a web or mobile app** to process real-world, user-supplied images. The pipeline will accept any user-supplied image as input and will predict whether a dog or human is present in the image. If a dog is detected in the image, it will provide an estimate of the dog’s breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 


## Problem Statement<a name="statement"></a>
In this project, I am provided with RGB images of humans and dogs and asked to design and implement an algorithm that can detect humans (human detector) or dogs (dog detector) in the images. After detecting a human or dog, the algorithm further needs to predict the breed of the dog (if the dog is detected) and the most resembling dog breed (if a human is detected). If neither is detected in the image, the algorithm should ask the user to input another image containing either dog or human.


## Data Exploration and Visualization<a name="eda"></a>
The dog breed dataset contains 8351 dog images with 133 dog breed categories. The dataset is not perfectly balanced. The mean number of images in each class is around 50. But there are few classes in the dataset that have less than 30 images while there are some classes that have more than 70 images. This small imbalance in data could pose a problem in training the dog breed classifier model. But this could be taken care of by over-sampling the minority classes or under-sampling the majority classes and data augmentation methods.


## Human Detector<a name="human"></a>
I used the pre-trained Haar cascade face detector model from the OpenCV library to determine if a human is present in the image or not.

## Dog Detector<a name="dog"></a>
To detect the dogs in the images, I have used a pre-trained VGG16 model. This model has been trained on ImageNet, a very large and popular dataset used for image classification and other vision tasks.

## List of Dependencies<a name="dependency"></a>
The `requirements.txt` list all the libraries/dependencies required to run this project.

1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and prepare image label pairs for training the model.

2. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and prepare images for the face detector model.


## File Descriptions<a name="desc"></a>
1. The `haarcascades folder` contains the pre-trained weights in the `xml file format` to use with the OpenCv face detector class that has been used in this project. 

2. The `results folder` contains the results of the algorithm tested on the test images. These are used for the purpose of quick demonstration in the results section below.

3. The `dog_app.ipynb file` is the main file for this project. It is a jupyter notebook containing code of face detector, dog detector and dog breed classifier models. The final algorithm that uses all these three models to make predictions is also implemented in this notebook. This notebook uses PyTorch to train all the models.

4. The `EfficientNetB4.ipynb` contains the code to train a dog breed classifier using TensorFlow on Google Colab. This model achieved an accuracy of 90.75% on the test dataset, average f1-score of 89.3%, precision score of 91.3% and recall score of 89.4%.


## Conclusion<a name="conc"></a>
This project serves as a good starting point to enter into the domain of deep learning. Data exploration and visualizations are extremely important before training any Machine Learning model as it helps in choosing a suitable performance metric for evaluating the model. CNN models in TensorFlow/PyTorch needs image data in the form of a 4D tensor. All images need to be reshaped into the same shape for training the CNN models in batch. 

Building CNN models from scratch is extremely simple in PyTorch. But training CNN models from scratch is computationally expensive and time-consuming. There are many pre-trained models available in PyTorch/TensorFlow (trained on ImageNet dataset) that can be used for transfer learning.

The most interesting thing to note is the power of transfer learning to achieve good results with small computation. It works well when the task is similar to the task on which the pre-trained model weights are optimized.


## Tips to improve the performance<a name="improve"></a>
1. Get more images per class
2. Make the dataset balanced
3. Use image augmentation methods such as CutOut, MixUp, and CutMix
4. Use VAEs/GANs to generate artificial data
5. Use activation maps to interpret the model predictions
6. Use deep learning-based approaches to detect human faces (MTCNN)


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to Udacity for the data and python 3 notebook.




