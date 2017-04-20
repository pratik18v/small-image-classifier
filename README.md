# Deep Learning based Image Classifier

## Dataset Overview
The dataset used is a subset of the Imagenet dataset. It contains images for 3 classes - dog, cat, human. The number of images per class are as follows:

* Class-1 (Dog): 894
* Class-2 (Cat): 1132
* Class-3 (Human): 874

Total number of images are 2900.

I have used some data augmentation techniques to increase the size of the dataset. They are as follows:

* Flip the image about the vertical axis.
* Add gaussian noise.
* Increase the contrast of the image.
* Crop the image by 10% from all 4 edges.
* Rotate the image by 20-degrees.

The new dataset has the following number of images:

* Class-1 (Dog): 5364
* Class-2 (Cat): 6792
* Class-3 (Human): 5244

Total number of images are 17400.

The number of images required to build a robust classifier is a function of the complexity of the model we're trying to train. The more complex the model (i.e., the more parameters it learns), the more data it requires. For example, a simple classifier like SVM will require less data to perform good enough as comapred to a multi-layered CNN.

## Models
I have experimented with the following models:

1. **Basic CNN**: This model consists of the following layers:
    * Layer 1: A convolution layer with kernel size: 5 x 5 x 32.
    * Layer 2: A max-pooling layer with downscale factor of 2.
    * Layer 3: A convolution layer with kernel size: 5 x 5 x 64.
    * Layer 4: A max-pooling layer with downscale factor of 2.
    * Layer 5: A dense layer with 1024 hidden units.
    * Layer 6: Soft-max layer with 3 nodes (i.e. the number of classes).
    
2. **Wide Residual Network**: The exact details of this architecture are described in the paper ([Link](https://arxiv.org/pdf/1605.07146.pdf)).

## Results
Image size used: 32 x 32 x 3 <br />

* **Model: Basic CNN** <br />
Batch size: 100 <br />
Learning rate: 0.01 <br />
Iterations: 20,000 <br />
Drop-out probability: 0.5 (training), 1.0 (testing) <br />
Test accuracy: 80%

* **Model: Wide Residual Network** <br />
    
<table>
  <tr>
    <th>S.R. No.</th>
    <th>N/w Width (k)</th>
    <th>Units per block (n)</th>
    <th>Learning Rate </th>
    <th>Batch Size</th>
    <th>Iterations</th>
    <th>Dropout Prob</th>
    <th>Test Acc (%)</th>
  </tr>
  <tr>
    <td>1</td>
    <td>1</td>
    <td>2</td>
    <td>0.01</td>
    <td>100</td>
    <td>50,000</td>
    <td>0.5</td>
    <td>90</td>
  </tr>
  <tr>
    <td>2</td>
    <td>1</td>
    <td>3</td>
    <td>0.01</td>
    <td>100</td>
    <td>50,000</td>
    <td>0.5</td>
    <td>93</td>
  </tr>  
  <tr>
    <td>3</td>
    <td>2</td>
    <td>2</td>
    <td>0.01</td>
    <td>100</td>
    <td>50,000</td>
    <td>0.5</td>
    <td>94</td>
  </tr>  
  <tr>
    <td>4</td>
    <td>2</td>
    <td>3</td>
    <td>0.01</td>
    <td>100</td>
    <td>50,000</td>
    <td>0.5</td>
    <td>93.8</td>
  </tr>
  <tr>
    <td>5</td>
    <td>2</td>
    <td>2</td>
    <td>0.01</td>
    <td>100</td>
    <td>50,000</td>
    <td>0.3</td>
    <td>94.2</td>
  </tr>
   <tr>
    <td>6</td>
    <td>2</td>
    <td>2</td>
    <td>0.01</td>
    <td>200</td>
    <td>50,000</td>
    <td>0.3</td>
    <td>94.8</td>
  </tr>
   <tr>
    <td>7</td>
    <td>2</td>
    <td>3</td>
    <td>0.01</td>
    <td>200</td>
    <td>50,000</td>
    <td>0.3</td>
    <td>95.36</td>
  </tr>
    <tr>
    <td>8</td>
    <td>3</td>
    <td>3</td>
    <td>0.01</td>
    <td>200</td>
    <td>100,000</td>
    <td>0.3</td>
    <td><b>95.90</b></td>
  </tr>
</table>

## Further Improvements
We can improve the performance further by using transfer learning in the following 2 ways - 
* Cut-off the final layer of a model pre-trainned on similar kind of classes. Attach a new softmax layer for our 3 classes and start training this new model on our data. This is called fine-tuning.

* Pass the images through a pre-trainned model and extract features from the second-last layer. Use these features with some other classifier like SVM or xgboost (boosting with decision trees). In case of xgboost we will have to train 3 one-vs-all classifiers and take a weighted combination of their outputs at the time of testing.

With both the above methods we can train a very good performance classifier using relatively less amount of data and much faster as well.

We can also experiment with ensemble networks to see if the performance can be improved.
