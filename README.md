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

## 3. Forward Propagation

![alt text](https://github.com/jingieboy/Neural_Network_From_Scratch/blob/main/ignore_img/forward_prop.png)

- $A^{[0]} = X$ is our input layer, there is no processing there, it is just the 784 pixels.
- $Z^{[1]} = W^{[1]} A^{[0]} + b^{[1]}$ calculates the linear transformation for the first hidden layer, and introduces weights and biases to the equations.
- $A^{[1]} = g(Z^{[1]}) = ReLU(Z^{[1]})$ applies the ReLU activation function to introduce non-linearity.
- $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$ computes the linear transformation for the output layer, with weights and biases from the first hidden layer.
- $A^{[2]} = softmax(Z^{[2]})$ applies the softmax activation function to produce class probabilities.

## 4. Backward Propagation

![alt text](https://github.com/jingieboy/Neural_Network_From_Scratch/blob/main/ignore_img/backward_prop.png)

- $dZ^{[2]} = A^{[2]} - Y$ calculates the derivative of the cost with respect to $Z^{[2]}$.
- $db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$ computes the gradient of the bias for layer 2.
- $dZ^{[1]} = W^{[2]T} dZ^{[2]} \cdot g'(Z^{[1]})$ calculates the derivative of the cost with respect to $Z^{[1]}$ using the chain rule.
- $dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T$ computes the gradient of the weights for layer 1.
- $db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$ calculates the gradient of the bias for layer 1.

## 5. Updating Parameters after Gradient Descent

![alt text](https://github.com/jingieboy/Neural_Network_From_Scratch/blob/main/ignore_img/params.png)

- $W^{[1]} = W^{[1]} - \alpha dW^{[1]}$ updates the weights for layer 1 using the learning rate $\alpha$ and the gradient $dW^{[1]}$.
- $b^{[1]} = b^{[1]} - \alpha db^{[1]}$ updates the bias for layer 1.
- $W^{[2]} = W^{[2]} - \alpha dW^{[2]}$ updates the weights for layer 2.
- $b^{[2]} = b^{[2]} - \alpha db^{[2]}$ updates the bias for layer 2.






