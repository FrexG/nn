#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

#define NUM_SAMPLES 4
#define NUM_INPUTS 2

int main(void) {
  srand(time(0));

  // Create a neural network
  size_t layer_sizes[] = {2,2,1};
  size_t count = ARRAY_SIZE(layer_sizes);
  ActivationFunction activations[] = {relu,sigmoid};

  Tensor in = new_tensor(1,*layer_sizes);
  rand_tensor(in,-1,1);

  Net neural_net = fully_connected_layer(layer_sizes,count,activations);
  print_network(neural_net,"ff");

  printf("Performing forward pass ... \n");
  _forward(neural_net,in);
  print_network(neural_net,"ff");

  /* an Xor dataset */
  float xor_input[NUM_SAMPLES][NUM_INPUTS] = {{0,0},{0,1},{1,0},{1,1}};
  float xor_output[NUM_SAMPLES] = {0,1,1,0};
  float* xor_out_mem = calloc(NUM_SAMPLES,sizeof(float));

  for(int i = 0; i < NUM_SAMPLES; ++i){
    xor_out_mem[i] = xor_output[i];
  }

  // training loop
  Tensor xor_input_tensor[] = {};
  Tensor xor_output_tensor = new_tensor(NUM_SAMPLES,1);
  set_tensor(xor_output_tensor,NUM_SAMPLES,1,xor_out_mem);
  free(xor_out_mem);
  PRINT_T(xor_output_tensor);

  for(size_t i = 0; i < NUM_SAMPLES; ++i){
      // print sub tensors
      Tensor temp_tensor = new_tensor(1,NUM_INPUTS);
      float *temp_holder = calloc(NUM_INPUTS,sizeof(float)); 

      for(size_t j = 0; j < NUM_INPUTS;++j){
        temp_holder[j] = xor_input[i][j];
      }
      set_tensor(temp_tensor,1,NUM_INPUTS,temp_holder);
      free(temp_holder);
  }
  free_neural_net(neural_net);
   return 0;
  
}
