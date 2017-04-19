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
    <td><b>95.36</b></td>
  </tr>
</table>
