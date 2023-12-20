#include <time.h>
#define NN_IMPLEMENTATION

#include "../../nn.h"
#define NUM_SAMPLES 4
#define NUM_INPUTS 2
#define NUM_OUTPUTS 1

void train(Net neural_net, Tensor x_train, Tensor y_train, size_t epochs, float lr)
{
  // train loop
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

      _forward(neural_net, x);
      // backprop
      _backward(neural_net, y);
      free(y.es);
      free(x.es);
    }
    _update(neural_net, lr, NUM_SAMPLES);
    zero_grad(neural_net);
  }
}
void evaluate(Net neural_net, Tensor x_train, Tensor y_train)
{
  // evaluation
  printf("Evaluate...\n");
  float cost = 0.0;
  for (size_t i = NUM_SAMPLES; i > 0; --i)
  {
    Tensor x = new_tensor(1, NUM_INPUTS);
    Tensor y = new_tensor(1, NUM_OUTPUTS);

    float temp_vals[1][NUM_INPUTS];
    for (size_t j = 0; j < NUM_INPUTS; ++j)
    {
      temp_vals[0][j] = VALUE_AT(x_train, i - 1, j);
    }
    set_tensor(x, 1, NUM_INPUTS, temp_vals);
    VALUE_AT(y, 0, 0) = VALUE_AT(y_train, i - 1, 0);

    Tensor pred = _forward(neural_net, x);
    printf("%f ^ %f  = %f\n", *x.es, *(x.es + 1), *pred.es);
    Tensor cost_t = mse(y, pred);
    cost += *cost_t.es;
    free(cost_t.es);
    free(y.es);
    free(x.es);
  }
  printf("Cost = %f\n", cost / NUM_SAMPLES);
}

int main(void)
{
  // srand(time(0));
  srand(69);

  // Create a neural network
  size_t layer_sizes[] = {2, 2, 1};
  size_t count = ARRAY_SIZE(layer_sizes);

  Activation activations[] = {RELU, SIGMOID};

  Net neural_net = fully_connected_layer(layer_sizes, count, activations);
  printf("Network initial parameters ... \n");
  // print_network(neural_net, "ff");

  /* an Xor dataset */
  float xor_input[NUM_SAMPLES][NUM_INPUTS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  float xor_output[NUM_SAMPLES][NUM_OUTPUTS] = {{0}, {1}, {1}, {0}};

  Tensor xor_input_tensor = new_tensor(NUM_SAMPLES, NUM_INPUTS);
  set_tensor(xor_input_tensor, NUM_SAMPLES, NUM_INPUTS, xor_input);

  Tensor xor_output_tensor = new_tensor(NUM_SAMPLES, 1);
  set_tensor(xor_output_tensor, NUM_SAMPLES, 1, xor_output);

  PRINT_T(xor_input_tensor);
  PRINT_T(xor_output_tensor);

  // start training
  train(neural_net, xor_input_tensor, xor_output_tensor, 100 * 1000, 0.1);
  // evaluat
  evaluate(neural_net, xor_input_tensor, xor_output_tensor);

  free_neural_net(&neural_net);
  free(xor_input_tensor.es);
  free(xor_output_tensor.es);
  return 0;
}
