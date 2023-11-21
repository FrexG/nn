#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"


int main(void) {
  srand(time(0));

  // Create a neural network
  size_t layer_sizes[] = {2,2,1};
  size_t count = ARRAY_SIZE(layer_sizes);
  ActivationFunction activations[] = {relu,sigmoid};

  Net neural_net = fully_connected_layer(layer_sizes,count,activations);
  print_network(neural_net,"ff");

  printf("Performing forward pass ... \n");
  _forward(neural_net);
  print_network(neural_net,"ff");
  
  return 0;
  
}
