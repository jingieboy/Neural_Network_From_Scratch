# Neural Network from Scratch (Only Numpy & Math)

# Hand-Written Digit Classifier

![alt text](https://github.com/jingieboy/Neural_Network_From_Scratch/blob/main/ignore_img/MNIST_dataset_example.png)

Here is a simple neural network architecture to recognise hand-written digits based of the famous MNIST dataset

## 1. About the dataset

![alt text](https://github.com/jingieboy/Neural_Network_From_Scratch/blob/main/ignore_img/goal.png)

The training image are 28 x 28 pixels, 784 pixels in total. Since images are greyscaled, each pixel ranges from 0 to 255. 255 means white, 0 means black. 

## 2. Neural Network Architecture

![alt text](https://github.com/jingieboy/Neural_Network_From_Scratch/blob/main/ignore_img/architecture.png)

This is a simple neural network, only with 2 layers.
1. **Input Layer:** Contains our 784 nodes, each mapped to a node
2. **Hidden Layer:** 10 units, ReLU activation function
3. **Output Layer:** 10 output units (each representing 1 digit), Softmax activation function


