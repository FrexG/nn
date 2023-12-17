
#ifndef NN_H_
#define NN_H_

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PRINT_T(m) print_tensor(m, #m)
#define VALUE_AT(t, i, j) (t).es[(i) * (t).cols + (j)]
#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))

#define RELU_PARAM 0.01f

typedef struct
{
  /*define a matrix
   * with arbitary size
   */
  size_t rows;
  size_t cols;
  float *es; // pointer to the memory that hold the float(values)
} Tensor;

typedef enum
{
  SIGMOID,
  RELU
} Activation;

typedef struct
{
  size_t net_size;
  Tensor *weights; // array of weights for the network
  Tensor *biases;  // array of biases for the network
  Tensor *grad_weights;
  Tensor *grad_biases;
  Tensor *activations; // outputs from each layer
  Tensor *grad_activations;
  Activation *activation_funs;
} Net;

Tensor new_tensor(size_t, size_t);

float float_rand(void);
float _sigmoid(float val);
float _relu(float val);
Tensor mse(Tensor y_true, Tensor y_pred);
// Tensor initializations
void rand_tensor(Tensor src, int min, int max);
void zeros_tensor(Tensor src);
void ones_tensor(Tensor src);
void set_tensor(Tensor src, int rows, int cols, float (*val)[cols]);
void copy_tensor(Tensor dst, Tensor src);
// Tensor operation
void matmul(Tensor dst, Tensor a, Tensor b);
void matadd(Tensor dst, Tensor src);
void matdiff(Tensor dst, Tensor src);
void threshold(Tensor src, float T);
// derivatives of activation function

float d_act_fun(float val, Activation act_fun);
void zero_grad(Net nn);
// Basic nn
Net fully_connected_layer(size_t *l_sizes, size_t count,
                          Activation *layer_activations);

Tensor _forward(Net nn, Tensor input);
void _backward(Net nn, Tensor cost);
void _update(Net nn, float lr, size_t batch);
void normalize(Tensor grad, size_t batch);
void grad_diff(Tensor param, Tensor grad, float lr);

void print_tensor(Tensor t, char *name);
void print_network(Net nn, char *name);

void free_neural_net(Net *);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float float_rand(void) { return (float)rand() / (float)RAND_MAX; }

Tensor new_tensor(size_t rows, size_t cols)
{
  // allocate an object of type Tensor on the stack
  Tensor m;
  m.rows = rows;
  m.cols = cols;
  // calloc so values will be initialized to zero by default
  m.es = calloc(rows * cols, sizeof(*m.es));
  assert(m.es != NULL);
  return m;
}

void rand_tensor(Tensor m, int min, int max)
{
  for (size_t i = 0; i < m.rows; ++i)
  {
    for (size_t j = 0; j < m.cols; ++j)
    {
      VALUE_AT(m, i, j) = float_rand() * (max - min) + min;
    }
  }
  return;
}
void zeros_tensor(Tensor m)
{
  for (size_t i = 0; i < m.rows; ++i)
  {
    for (size_t j = 0; j < m.cols; ++j)
    {
      VALUE_AT(m, i, j) = 0.0f;
    }
  }
  return;
}
void ones_tensor(Tensor m)
{
  for (size_t i = 0; i < m.rows; ++i)
  {
    for (size_t j = 0; j < m.cols; ++j)
    {
      VALUE_AT(m, i, j) = 1.0;
    }
  }
  return;
}
void set_tensor(Tensor m, int rows, int cols, float (*val)[cols])
{
  // set the value
  for (size_t i = 0; i < m.rows; ++i)
  {
    for (size_t j = 0; j < m.cols; ++j)
    {
      VALUE_AT(m, i, j) = val[i][j];
    }
  }
}
void copy_tensor(Tensor dst, Tensor src)
{
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; ++i)
  {
    for (size_t j = 0; j < dst.cols; ++j)
    {
      VALUE_AT(dst, i, j) = VALUE_AT(src, i, j);
    }
  }
}
void matmul(Tensor dst, Tensor a, Tensor b)
{
  /* Tensor multiplication between to matrices
   * Args:
   *  a(Tensor): firs matrice
   *  b(Tensor): second matrice
   * a.cols == b.rows
   * Returns:
   *  Tensor
   * */
  assert(a.cols == b.rows);
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  // multiply ith row of matrice a with jth col of matrice b
  for (size_t i = 0; i < dst.rows; ++i)
  {
    for (size_t j = 0; j < dst.cols; ++j)
    {
      VALUE_AT(dst, i, j) = 0;
      for (size_t k = 0; k < a.cols; ++k)
      {
        VALUE_AT(dst, i, j) += VALUE_AT(a, i, k) * VALUE_AT(b, k, j);
      }
    }
  }
}
void matadd(Tensor dst, Tensor src)
{
  /* Elementwise summation of two matrices*/
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);

  for (size_t i = 0; i < dst.rows; ++i)
  {
    for (size_t j = 0; j < dst.cols; ++j)
    {
      VALUE_AT(dst, i, j) += VALUE_AT(src, i, j);
    }
  }
}
void matdiff(Tensor dst, Tensor src)
{
  /* Elementwise summation of two matrices*/
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);

  for (size_t i = 0; i < dst.rows; ++i)
  {
    for (size_t j = 0; j < dst.cols; ++j)
    {
      VALUE_AT(dst, i, j) -= VALUE_AT(src, i, j);
    }
  }
}

/*Operations*/
float _sigmoid(float x)
{
  return 1 / (1 + expf(-x));
}
float _relu(float x)
{
  return x > 0 ? x : x * RELU_PARAM;
}

float activate(float val, Activation activation_type)
{
  switch (activation_type)
  {
  case SIGMOID:
    return _sigmoid(val);
    break;
  case RELU:
    return _relu(val);
    break;
  }
}
float d_act_fun(float val, Activation act_fun)
{
  switch (act_fun)
  {
  case SIGMOID:
    return val * (1 - val);
    break;
  case RELU:
    return val >= 0 ? 1 : RELU_PARAM;
    break;
  }
  return 0;
}
void threshold(Tensor src, float T)
{
  for (size_t i = 0; i < src.rows; ++i)
  {
    for (size_t j = 0; j < src.cols; ++j)
    {
      VALUE_AT(src, i, j) = VALUE_AT(src, i, j) > T ? 1. : 0.;
    }
  }
}
Tensor mse(Tensor y_true, Tensor y_pred)
{
  Tensor cost_t = new_tensor(y_true.rows, y_true.cols);
  for (size_t i = 0; i < y_true.rows; ++i)
  {
    for (size_t j = 0; j < y_true.cols; ++j)
    {
      VALUE_AT(cost_t, i, j) =
          pow((VALUE_AT(y_pred, i, j) - VALUE_AT(y_true, i, j)), 2);
    }
  }
  return cost_t;
}
void zero_grad(Net nn)
{
  for (size_t i = 0; i < nn.net_size; ++i)
  {
    zeros_tensor(nn.grad_activations[0]);
    zeros_tensor(nn.grad_weights[i]);
    zeros_tensor(nn.grad_biases[i]);
  }
  zeros_tensor(nn.grad_activations[nn.net_size]);
}
Net fully_connected_layer(size_t *l_sizes, size_t count,
                          Activation *layer_activations)
{
  size_t hidden_layer_count = count - 1;
  // initialize
  Net nn = {.net_size = hidden_layer_count,
            .weights = calloc(hidden_layer_count, sizeof(Tensor)),
            .biases = calloc(hidden_layer_count, sizeof(Tensor)),
            .grad_weights = calloc(hidden_layer_count, sizeof(Tensor)),
            .grad_biases = calloc(hidden_layer_count, sizeof(Tensor)),
            .activations = calloc(count, sizeof(Tensor)),
            .grad_activations = calloc(count, sizeof(Tensor)),
            .activation_funs = layer_activations};
  /*initialize each tensors in weights & biases*/
  // initialize the input activation, it will have a row 1, and col of
  // l_sizes[0]
  nn.activations[0] = new_tensor(1, l_sizes[0]);
  nn.grad_activations[0] = new_tensor(1, l_sizes[0]);
  // initialize weights
  for (size_t i = 0; i < hidden_layer_count; ++i)
  {
    // weights
    nn.weights[i] = new_tensor(nn.activations[i].cols, l_sizes[i + 1]);
    nn.biases[i] = new_tensor(1, l_sizes[i + 1]);
    // randomize weights and biases
    rand_tensor(nn.weights[i], -1, 1);
    rand_tensor(nn.biases[i], -1, 1);
    nn.grad_weights[i] = new_tensor(nn.activations[i].cols, l_sizes[i + 1]);
    nn.grad_biases[i] = new_tensor(1, l_sizes[i + 1]);
    // zero grads
    zeros_tensor(nn.grad_weights[i]);
    zeros_tensor(nn.grad_biases[i]);
    // next_activation
    nn.activations[i + 1] = new_tensor(1, l_sizes[i + 1]);
    nn.grad_activations[i + 1] = new_tensor(1, l_sizes[i + 1]);
  }

  return nn;
}
Tensor _forward(Net nn, Tensor input)
{
  // Perform matmul between input and weight
  // add bias
  // activation function
  // set the initial activation to the value fo the input tenosr
  // free already alocated tensor memory
  // free(nn.activations[0].es);
  // copy input tensor to nn input
  copy_tensor(nn.activations[0], input);

  for (size_t l = 0; l < nn.net_size; ++l)
  {
    matmul(nn.activations[l + 1], nn.activations[l], nn.weights[l]);
    matadd(nn.activations[l + 1], nn.biases[l]);
    // activate each layers activation with an the layers activation functioin
    size_t rows = nn.activations[l + 1].rows;
    size_t cols = nn.activations[l + 1].cols;

    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t j = 0; j < cols; ++j)
      {
        float val = VALUE_AT(nn.activations[l + 1], i, j);
        VALUE_AT(nn.activations[l + 1], i, j) = activate(val, nn.activation_funs[l]);
      }
    }
  }
  return nn.activations[nn.net_size]; // last activation is output
}
void _backward(Net nn, Tensor target)
{

  for (size_t l = 0; l <= nn.net_size; ++l)
  {
    zeros_tensor(nn.grad_activations[l]);
  }

  for (size_t j = 0; j < target.cols; ++j)
  {
    VALUE_AT(nn.grad_activations[nn.net_size], 0, j) = 2 * (VALUE_AT(nn.activations[nn.net_size], 0, j) - VALUE_AT(target, 0, j));
  }
  for (size_t l = nn.net_size; l > 0; --l)
  {
    for (size_t i = 0; i < nn.activations[l].cols; ++i)
    {
      float _act = VALUE_AT(nn.activations[l], 0, i);
      float _e = VALUE_AT(nn.grad_activations[l], 0, i);

      float _d_act = d_act_fun(_act, nn.activation_funs[l - 1]);
      // gradient of the bias
      VALUE_AT(nn.grad_biases[l - 1], 0, i) += 2 * _e * _d_act;

      for (size_t k = 0; k < nn.activations[l - 1].cols; ++k)
      {
        float _prev_act = VALUE_AT(nn.activations[l - 1], 0, k);
        // printf("_prev_act = %f\n", _prev_act);

        float _w = VALUE_AT(nn.weights[l - 1], k, i);
        // calculate the gradients

        VALUE_AT(nn.grad_weights[l - 1], k, i) += 2 * _e * _d_act * _prev_act;
        VALUE_AT(nn.grad_activations[l - 1], 0, k) += 2 * _e * _d_act * _w;
      }
    }
  }
}
void _update(Net nn, float lr, size_t batch)
{
  for (size_t n = 0; n < nn.net_size; ++n)
  {
    normalize(nn.grad_weights[n], batch);
    normalize(nn.grad_biases[n], batch);
    // update the weights
    grad_diff(nn.weights[n], nn.grad_weights[n], lr);
    grad_diff(nn.biases[n], nn.grad_biases[n], lr);
  }
}
void normalize(Tensor grad, size_t batch)
{
  for (size_t i = 0; i < grad.rows; ++i)
  {
    for (size_t j = 0; j < grad.cols; ++j)
    {
      VALUE_AT(grad, i, j) /= batch;
    }
  }
}
void grad_diff(Tensor param, Tensor grad, float lr)
{
  assert(param.rows == grad.rows);
  assert(param.cols == grad.cols);

  for (size_t i = 0; i < param.rows; ++i)
  {
    for (size_t j = 0; j < param.cols; ++j)
    {
      VALUE_AT(param, i, j) -= lr * VALUE_AT(grad, i, j);
    }
  }
}
void free_neural_net(Net *nn)
{
  free(nn->activations[nn->net_size].es);
  free(nn->grad_activations[nn->net_size].es);

  for (size_t i = 0; i < nn->net_size; ++i)
  {
    free(nn->weights[i].es);
    free(nn->biases[i].es);
    free(nn->grad_weights[i].es);
    free(nn->grad_biases[i].es);
    free(nn->activations[i].es);
    free(nn->grad_activations[i].es);
  }
  free(nn->weights);
  free(nn->biases);
  free(nn->grad_weights);
  free(nn->grad_biases);
  free(nn->activations);
  free(nn->grad_activations);
}
void print_tensor(Tensor m, char *name)
{
  printf("%s = [\n", name);
  for (size_t i = 0; i < m.rows; ++i)
  {
    printf("\t[");
    for (size_t j = 0; j < m.cols; j++)
    {
      printf("%f,", VALUE_AT(m, i, j));
    }
    printf("],\n");
  }
  printf("]\n");
  return;
}
void print_network(Net nn, char *name)
{
  printf("Network Summary for network %s.... \n", name);
  char buf[256];
  print_tensor(nn.activations[0], "Input");
  for (size_t i = 0; i < nn.net_size; ++i)
  {
    sprintf(buf, "w_%lu", i);
    print_tensor(nn.weights[i], buf);

    sprintf(buf, "gw_%lu", i);
    print_tensor(nn.grad_weights[i], buf);

    sprintf(buf, "b_%lu", i);
    print_tensor(nn.biases[i], buf);

    sprintf(buf, "gb_%lu", i);
    print_tensor(nn.grad_biases[i], buf);

    sprintf(buf, "a_%lu", i + 1);
    print_tensor(nn.activations[i + 1], buf);

    sprintf(buf, "ga_%lu", i + 1);
    print_tensor(nn.grad_activations[i + 1], buf);
  }
}
#endif // NN_IMPLEMENTATION