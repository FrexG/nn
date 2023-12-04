#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

#define NUM_SAMPLES 4
#define NUM_INPUTS 2
#define NUM_OUTPUTS 1

int main(void) {
  srand(time(0));

  // Create a neural network
  size_t layer_sizes[] = {2,2,1};
  size_t count = ARRAY_SIZE(layer_sizes);
  ActivationFunction activations[] = {relu,sigmoid};

  //Tensor in = new_tensor(1,*layer_sizes);
  //rand_tensor(in,-1,1);

  Net neural_net = fully_connected_layer(layer_sizes,count,activations);
  print_network(neural_net,"ff");

  printf("Performing forward pass ... \n");
  //Tensor pred = _forward(neural_net,in);
  //print_network(neural_net,"ff");

  /* an Xor dataset */
  float xor_input[NUM_SAMPLES][NUM_INPUTS] = {{0,0},{0,1},{1,0},{1,1}};
  float xor_output[NUM_SAMPLES][NUM_OUTPUTS] = {{0},{1},{1},{0}};

  Tensor xor_input_tensor = new_tensor(NUM_SAMPLES,NUM_INPUTS);
  set_tensor(xor_input_tensor,NUM_SAMPLES,NUM_INPUTS,xor_input);

  Tensor xor_output_tensor = new_tensor(NUM_SAMPLES,1);
  set_tensor(xor_output_tensor,NUM_SAMPLES,1,xor_output);

  PRINT_T(xor_input_tensor);
  PRINT_T(xor_output_tensor);

  // train loop
  float cost = 0.0;

  for(size_t i = 0; i < NUM_SAMPLES; ++i){
    Tensor temp_tensor = new_tensor(1,NUM_INPUTS);
    float temp_vals[1][NUM_INPUTS];
    for(size_t j = 0; j < NUM_INPUTS; ++j){
      temp_vals[0][j] = VALUE_AT(xor_input_tensor,i,j);
    }
    set_tensor(temp_tensor,1,NUM_INPUTS,temp_vals);
    Tensor pred = _forward(neural_net,temp_tensor);
    cost += mse(VALUE_AT(xor_output_tensor,i,0),VALUE_AT(pred,0,0));
  }
  printf("Total cost = %f\n",cost);

  free_neural_net(&neural_net);
  free(xor_input_tensor.es);
  free(xor_output_tensor.es);
  return 0;
  
}
