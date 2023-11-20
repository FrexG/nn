#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

#define PRINT_T(m) print_tensor(m, #m)

int main(void) {
  srand(time(0));
  // Define data, xOR gate
  Tensor x = new_tensor(4,2);
  float x_val[4][2] = {
    {0,0},{0,1},{1,0},{1,1},
  };
  set_tensor(x,4,2,x_val);
  PRINT_T(x);

  Tensor y = new_tensor(4,1);
  float y_val[][1] = {
    {0},{1},{1},{0}
  };
  set_tensor(y,4,1,y_val);
  PRINT_T(y);

  // construct model
  int input_size = 2;
  int output_size = 1;
  int num_h_layers = 1; // one hidden layer(relu)
  int h_layers_s[] = {2};

  Tensor in_tensor = new_tensor(1, input_size);
  zeros_tensor(in_tensor);

  PRINT_T(in_tensor);
  // construct the model 
  Tensor *model=
      linear_layer(input_size, output_size, num_h_layers, h_layers_s);

  for (int i = 0; i < num_h_layers + 1; ++i) {
    PRINT_T(model[i]);
  }

  ActivationFunction layer_activations[] = {relu,sigmoid};

  printf("Performing forward pass .... \n");

  Tensor logits =
      feed_forward(num_h_layers + 1, in_tensor, model, layer_activations);

  PRINT_T(logits);
  return 0;
}
