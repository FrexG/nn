# nn.h

### A Simple Tensor/Neural Network library with autograd (backpropagation) in C

`nn.h` is a lightweight and easy-to-use neural network library designed to provide a PyTorch-like API for training neural networks in C.
The library focuses on basic array/tensor operations, supporting essential functionalities such as `matrix multiplication`, `addition`, `initialization with ones or zeros`, `hresholding`, and `subtraction`. 
You can easily define arbitrary-length neural networks using this library.

## Features

- **Matrix Operations:** Perform matrix multiplication and basic array operations.
- **Initialization:** Initialize tensors with ones, zeros or random values for convenience.
- **Activation Functions:** Current version supports `sigmoid` and `relu` activation functions.
- **Autograd (Backpropagation):** Implement automatic differentiation for efficient backpropagation with a pytorch like api.

## Getting Started

### Usage

To use `nn.h` in your C project, simply include the single header file in your source code:

``` c
#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"
```
To create a new `Tensor`
``` c
/* Create a 2 by 4 tensor */
size_t rows = 2;
size_t cols = 4;
// by default new_tensor() initializes a Tensor with all zero values
Tensor tensor = new_tensor(rows,cols);
// to populate the tensor with random values
rand_tensor(tensor,-1,1); // will randomize elements of the tensor with values between -1 and 1
PRINT_T(tensor); // print the tensor
/*
  tensor = [
	[0.007344,-0.9232,,-0.6832,0.2345],
	[-0.207344,-0.7232,,-0.8642,0.16345],
]
*/
```
To create a `multi-layer` neural network
``` c
  // Create a neural network
  // create a neurla network with 2 hidden layers and 1 input layer
  // input layer accepts (1,2) inputs, first hidden layer shape = (1,2),
  // last hidden layer (output) shape = (1,1)
  size_t layer_sizes[] = {2, 2, 1};
  size_t count = ARRAY_SIZE(layer_sizes);
  // defien activation function of the hidden layers
  Activation activations[] = {RELU, SIGMOID};
  // initialize neural net
  Net neural_net = fully_connected_layer(layer_sizes, count, activations);
```
To train the network on some data
``` c
#define NUM_SAMPLES 4
#define NUM_INPUTS 2
#define NUM_OUTPUTS 1

int main(){
  // train loop
  size_t epochs = 100;
  float lr = 1e-3;
  /* an Xor dataset */
  float xor_input[NUM_SAMPLES][NUM_INPUTS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  float xor_output[NUM_SAMPLES][NUM_OUTPUTS] = {{0}, {1}, {1}, {0}};
  
  Tensor xor_input_tensor = new_tensor(NUM_SAMPLES, NUM_INPUTS);
  set_tensor(xor_input_tensor, NUM_SAMPLES, NUM_INPUTS, xor_input);
  
  Tensor xor_output_tensor = new_tensor(NUM_SAMPLES, 1);
  set_tensor(xor_output_tensor, NUM_SAMPLES, 1, xor_output);
  
  for (size_t epoch = 0; epoch < epochs; ++epoch)
  {
  
    for (size_t i = 0; i < NUM_SAMPLES; ++i)
    {
      Tensor x = new_tensor(1, NUM_INPUTS);
      Tensor y = new_tensor(1, NUM_OUTPUTS);
  
      float temp_vals[1][NUM_INPUTS];
      for (size_t j = 0; j < NUM_INPUTS; ++j)
      {
        temp_vals[0][j] = VALUE_AT(x_train, i, j);
      }
      set_tensor(x, 1, NUM_INPUTS, temp_vals);
      VALUE_AT(y, 0, 0) = VALUE_AT(y_train, i, 0);
      // forward pass
      _forward(neural_net, x);
      // backprop
      _backward(neural_net, y);
      free(y.es);
      free(x.es);
    }
    // update the weights
    _update(neural_net, lr, NUM_SAMPLES);
    // empty gradients
    zero_grad(neural_net);
}
}
```
### TODO
 - [ ] Batch processing
 - [ ] Softmax activation
 - [ ] Cross-entropy optimization

