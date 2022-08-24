## 🐶 Dog_Vision 🐶

## Table of Content
  * [Demo](#demo)
  * [Motivation](#motivation)
  * [Classify Different Dog Breeds](#classify-different-dog-breeds)
  * [Getting Data ready](#getting-data-ready)
  * [Creating and training the Neural Network](#creating-and-training-the-neural-network)
  * [Directory Tree](#directory-tree)
  * [Bug / Feature Request](#bug---feature-request)
  * [Future scope of project](#future-scope)


## Demo


### 🐶  Dog Breed Classification Project. 🐶 
An multiclass image classifier model using Tensorflow 2 and Tensorflow Hub to identify different breed of dogs

**You can see below  predictions made by the final model :**

<img src="https://user-images.githubusercontent.com/106836228/186512818-c9f84b27-bd15-4e12-aafc-adf799cdd48d.png">



## Motivation
What to do when you are at collage and having a strong base of mathematics and keen intrest in learning ML ,DL and AI? I started to learn Machine Learning model to get most out of it. I came to know mathematics behind models. Finally it is important to work on application (real world application) to actually make a difference.





## Classify Different Dog Breeds 

To do this, we'll be using data from the [Kaggle dog breed identification competition](https://www.kaggle.com/c/dog-breed-identification/overview). It consists of a collection of 10,000+ labelled images of 120 different dog breeds.

We're going to go through the following workflow:

#### 1. Problem

Identifying the breed of the dog from the image

#### 2.Data

The data we're using is from Kaggle's dog breed competition

#### 3. Evaluation

The evaluation is a file with prediction probabilities for each dog breed of each test image

#### 4. Features

Some information about the data:

- We're dealing with images( unstructured data ) so it's probably best if we use deep learning/transfer learning
- There are 120 breeds of dogs (120 different classes)
- There are around 10000 images in training(with labels) and test (without labels) set each


For preprocessing our data, we're going to use TensorFlow 2.x. The whole premise here is to get our data into Tensors (arrays of numbers which can be run on GPUs) and then allow a machine learning model to find patterns between them. For our machine learning model, we're gonna do some **transfer learning** and we're going to use a pretrained deep learning model from TensorFlow Hub. 

## Getting Data ready
### Preprocessing images (Turning images to Tensors)

To preprocess our images into Tensors we're going to :
1. Uses TensorFlow to read the file and save it to a variable, `image`.
2. Turn our `image` (a jpeg file) into Tensors.
3. Normalize image (from 0-255 to 0-1)
4. Resize the `image` to be of shape (224, 224).
5. Return the modified `image`.

A good place to read about this type of function is the [TensorFlow documentation on loading images](https://www.tensorflow.org/tutorials/load_data/images). 

### Creating data batches

Dealing with 10,000+ images may take up more memory than your GPU has. Trying to compute on them all would result in an error. So it's more efficient to create smaller batches of your data and compute on one batch at a time.

## Creating and training the Neural Network 
### mobilenet_v2_130_224 Model

In this project, we're using the **`mobilenet_v2_130_224`** model from TensorFlow Hub.
https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html
MobileNetV2 is a significant improvement over MobileNetV1 and pushes the state of the art for mobile visual recognition including classification, object detection and semantic segmentation. MobileNetV2 is released as part of TensorFlow-Slim Image Classification Library, or you can start exploring MobileNetV2 right away in Colaboratory. Alternately, you can download the notebook and explore it locally using Jupyter. MobileNetV2 is also available as modules on TF-Hub, and pretrained checkpoints can be found on github.
<img src="https://user-images.githubusercontent.com/106836228/186514899-12e8ca5a-0bb6-4c7d-a275-de8c064e1815.png">

### Setting up the model layers

The first layer we use is the model from TensorFlow Hub (`hub.KerasLayer(MODEL_URL)`. This **input layer** takes in our images and finds patterns in them based on the patterns [`mobilenet_v2_130_224`](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4) has found.

The next layer (`tf.keras.layers.Dense()`) is the **output layer** of our model. It brings all of the information discovered in the input layer together and outputs it in the shape we're after, 120 (the number of unique labels we have). The `activation="softmax"` parameter tells the output layer, we'd like to assign a probability value to each of the 120 labels [somewhere between 0 & 1](https://en.wikipedia.org/wiki/Softmax_function). The higher the value, the more the model believes the input image should have that label. 


## Performances and results


## Predicted Label/Breed and probability distribution
Another way to analyse the inference of our model is to print out the predicted class of an image and it's probability distribution. For each image, we show it's original label (left), it's predicted label (right), and the probabilty assossiated with the predicted label (how much confident is our Neural Network about the predicted class).

<img src="https://user-images.githubusercontent.com/106836228/186514589-6897e186-8260-4db1-b685-147abd367a82.png">


# Improve the model
How to approuve model accuracy :
1. [Trying another model from TensorFlow Hub](https://tfhub.dev/) - A different model could perform better on our dataset. 
2. [Data augmentation](https://bair.berkeley.edu/blog/2019/06/07/data_aug/) - Take the training images and manipulate (crop, resize) or distort them (flip, rotate) to create even more training data for the model to learn from. 
3. [Fine-tuning](https://www.tensorflow.org/hub/tf2_saved_model#fine-tuning)


## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)
 [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/)
 [<img target="_blank" src="https://user-images.githubusercontent.com/106836228/185730141-b8e739bb-8c0e-42ce-bc83-81f45cde875b.png" width=200>](https://www.tensorflow.org/)
 [<img target="_blank" src="https://user-images.githubusercontent.com/106836228/185730270-20281dad-529e-46b9-8a2d-385a6b46b32f.png" width=200>](https://www.tensorflow.org/api_docs/python/tf/keras)
 [<img target="_blank" src="https://user-images.githubusercontent.com/106836228/185730561-887e9dfb-1df8-4ed2-bea0-d3203eef2c49.png" width=200>](https://www.tensorflow.org/hub)
 
 
 
 ## Future Scope

* Use multiple Algorithms
* Deploying the model
* Front-End
