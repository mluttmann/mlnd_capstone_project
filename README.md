# Inventory Monitoring at Distribution Centers

The Amazon Fulfilment Centers are not built to store products only but also fulfil million orders from customers every day. In these distribution centers, robots are used to automatically track products while carrying them in bins. The process of collecting and packaging items by robots can be error prone. Therefore, checking the number of items in a bin is a good idea which can be solved with a trained model for image classification. This task is adressed by this repository and the results are described in the [report document](report/report.pdf).

## Project Set Up and Installation

For this project, an empty and accessibly S3 bucket has to be defined in the [Jupyter notebook](sagemaker.ipynb). The easiest way to use this notebook is to setup an AWS instance with Jupyter notebook loads the conda_pytorch_latest_p36 kernel. This kernel already includes all necessary Python libraries like the SageMaker SDK, PyTorch and torchvision.

## Dataset

### Overview
The dataset used in this project is the Amazon Bin Image Dataset that consists of more than 500,000 images of bins containing a different number of items. To reduce costs, a subset of this dataset is used which is defined in the [file_list.json](file_list.json) file.

### Access
Amazon Bin Image Dataset is publicly publicly available and is downloaded by the Jupyter notebook. More information on how to access the data can be found [here](Amazon Bin Image Dataset).

## Model Training
For transfer learning the pretrained model ResNet50 is used. This model is used with a cross-entropy loss function as well as an Adam optimizer which are chosen based on experience with similar image classification tasks and to get a better comparison with similiar projects.

## Machine Learning Pipeline
In this project, Amazon AWS and PyTorch is used to train machine learning model. The strategy to solve this problem consists of the following tasks:

* Download and classify a subset of the “Amazon Bin Image Dataset”.
* Split the dataset into parts for training, validation and testing and upload them into an Amazon S3 bucket.
* Perform transfer learning on a pretrained convolutional neural network called ResNet50. Optimal hyperparameters are found by tuning job beforehand.
* Investigate the achieved result by profiling.
* Deploy an end point and predict an example image.


## Standout Suggestions
* Hyperparameter tuning
* Debugging and Profiling
* Deployment of an end point
* Example prediction of an image
