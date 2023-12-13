
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

typedef struct {
  /*define a matrix
   * with arbitary size
   */
  size_t rows;
  size_t cols;
  float *es; // pointer to the memory that hold the float(values)
} Tensor;

typedef void (*ActivationFunction)(Tensor);

typedef struct {
  size_t net_size;
  Tensor *weights; // array of weights for the network
  Tensor *biases;  // array of biases for the network
  Tensor *grad_weights;
  Tensor *grad_biases;
  Tensor *activations; // outputs from each layer
  ActivationFunction *activation_funs;
} Net;

Tensor new_tensor(size_t, size_t);

float float_rand(void);
float _sigmoid(float);
Tensor mse(Tensor y_true, Tensor y_pred);
// Tensor initializations
void rand_tensor(Tensor src, int min, int max);
void zeros_tensor(Tensor src);
void ones_tensor(Tensor src);
void set_tensor(Tensor src, int rows, int cols, float (*val)[cols]);
// Tensor operation
void matmul(Tensor dst, Tensor a, Tensor b);
void matadd(Tensor dst, Tensor src);
void sigmoid(Tensor src);
void relu(Tensor src);
// derivatives of activation function
void d_sigmoid(Tensor activation, Tensor pre_activtion, Tensor grad,
               Tensor target, int is_weight);

void d_relu(Tensor activation, Tensor pre_activtion, Tensor grad, Tensor target,
            int is_weight);
// Basic nn
Net fully_connected_layer(size_t *l_sizes, size_t count,
                          ActivationFunction *layer_activations);

Tensor _forward(Net nn, Tensor input);
Tensor new_target(Tensor activation, Tensor target, Tensor pred, Tensor weight,
                  ActivationFunction act_fun);
void _backward(Net nn, Tensor cost);
void _update(Tensor param, Tensor grad);

void print_tensor(Tensor t, char *name);
void print_network(Net nn, char *name);

void free_neural_net(Net *);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float float_rand(void) { return (float)rand() / (float)RAND_MAX; }

Tensor new_tensor(size_t rows, size_t cols) {
  // allocate an object of type Tensor on the stack
  Tensor m;
  m.rows = rows;
  m.cols = cols;
  // calloc so values will be initialized to zero by default
  m.es = calloc(rows * cols, sizeof(*m.es));
  assert(m.es != NULL);
  return m;
}

void rand_tensor(Tensor m, int min, int max) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      VALUE_AT(m, i, j) = float_rand() * (max - min) + min;
    }
  }
  return;
}

void zeros_tensor(Tensor m) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      VALUE_AT(m, i, j) = 0.0f;
    }
  }
  return;
}
void ones_tensor(Tensor m) {
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      VALUE_AT(m, i, j) = 1.0;
    }
  }
  return;
}

void set_tensor(Tensor m, int rows, int cols, float (*val)[cols]) {
  // set the value
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      VALUE_AT(m, i, j) = val[i][j];
    }
  }
}

void copy_tensor(Tensor dst, Tensor src) {
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);

  for (size_t i = 0; i < src.rows; ++i) {
    for (size_t j = 0; j < src.cols; ++j) {
      VALUE_AT(dst, i, j) = VALUE_AT(src, i, j);
    }
  }
}

void matmul(Tensor dst, Tensor a, Tensor b) {
  /* Tensor multiplication between to matrices
   * Args:
   *  a(Tensor): firs matrice
   *  b(Tensor): second matrice
   * a.cols == b.rows
   * Returns:
   *  Tensor
   * */
  assert(a.cols == b.rows);
  // multiply ith row of matrice a with jth col of matrice b
  for (size_t i = 0; i < a.rows; ++i) {
    for (size_t j = 0; j < b.cols; ++j) {
      VALUE_AT(dst, i, j) = 0;
      for (size_t k = 0; k < a.rows; ++k) {
        VALUE_AT(dst, i, j) += VALUE_AT(a, i, k) * VALUE_AT(b, k, j);
      }
    }
  }
}

void matadd(Tensor dst, Tensor src) {
  /* Elementwise summation of two matrices
   * matrices should have the same dimension
   * Args:
   *  dst(Tensor): destination matrice
   *  src(Tensor): source matrice
   * */
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      VALUE_AT(dst, i, j) += VALUE_AT(src, i, j);
    }
  }
}

/*Operations*/

float _sigmoid(float x) { return 1 / (1 + expf(-x)); }

void sigmoid(Tensor src) {
  for (size_t i = 0; i < src.rows; ++i) {
    for (size_t j = 0; j < src.cols; ++j) {
      VALUE_AT(src, i, j) = _sigmoid(VALUE_AT(src, i, j));
    }
  }
}

void relu(Tensor src) {
  for (size_t i = 0; i < src.rows; ++i) {
    for (size_t j = 0; j < src.cols; ++j) {
      VALUE_AT(src, i, j) = fmaxf(0, VALUE_AT(src, i, j));
    }
  }
}

void d_sigmoid(Tensor activation, Tensor prev_activation, Tensor grad,
               Tensor target, int is_weight) {
  for (size_t j = 0; j < activation.rows; ++j) {
    for (size_t k = 0; k < activation.cols; ++k) {
      if (is_weight) {
        VALUE_AT(grad, j, k) =
            (VALUE_AT(activation, j, k) - VALUE_AT(target, j, k)) *
            (1 - VALUE_AT(activation, j, k)) * VALUE_AT(prev_activation, j, k);
      } else {
        VALUE_AT(grad, j, k) =
            (VALUE_AT(activation, j, k) - VALUE_AT(target, j, k)) *
            (1 - VALUE_AT(activation, j, k));
      }
    }
  }
}

void d_relu(Tensor activation, Tensor prev_activation, Tensor grad,
            Tensor target, int is_weight) {
  for (size_t j = 0; j < activation.rows; ++j) {
    for (size_t k = 0; k < activation.cols; ++k) {
      float d_relu_act = VALUE_AT(activation, j, k) >= 0 ? 1 : 0;
      if (is_weight) {
        VALUE_AT(grad, j, k) =
            (VALUE_AT(activation, j, k) - VALUE_AT(target, j, k)) * d_relu_act *
            VALUE_AT(prev_activation, j, k);
      } else {
        VALUE_AT(grad, j, k) =
            (VALUE_AT(activation, j, k) - VALUE_AT(target, j, k)) * d_relu_act;
      }
    }
  }
}

Net fully_connected_layer(size_t *l_sizes, size_t count,
                          ActivationFunction *layer_activations) {
  size_t hidden_layer_count = count - 1;
  // initialize
  Net nn = {.net_size = hidden_layer_count,
            .weights = calloc(hidden_layer_count, sizeof(Tensor)),
            .biases = calloc(hidden_layer_count, sizeof(Tensor)),
            .grad_weights = calloc(hidden_layer_count, sizeof(Tensor)),
            .grad_biases = calloc(hidden_layer_count, sizeof(Tensor)),
            .activations = calloc(count, sizeof(Tensor)),
            .activation_funs = layer_activations};
  /*initialize each tensors in weights & biases*/
  // initialize the input activation, it will have a row 1, and col of
  // l_sizes[0]
  nn.activations[0] = new_tensor(1, l_sizes[0]);
  // initialize weights
  for (size_t i = 0; i < hidden_layer_count; ++i) {
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
  }

  return nn;
}

Tensor _forward(Net nn, Tensor input) {
  // Perform matmul between input and weight
  // add bias
  // activation function
  // set the initial activation to the value fo the input tenosr
  // free already alocated tensor memory
  free(nn.activations[0].es);
  nn.activations[0] = input;
  for (size_t i = 0; i < nn.net_size; ++i) {
    matmul(nn.activations[i + 1], nn.activations[i], nn.weights[i]);
    matadd(nn.activations[i + 1], nn.biases[i]);
    nn.activation_funs[i](nn.activations[i + 1]);
  }
  return nn.activations[nn.net_size]; // last activation is output
}

Tensor mse(Tensor y_true, Tensor y_pred) {
  Tensor cost_t = new_tensor(y_true.rows, y_true.cols);
  for (size_t i = 0; i < y_true.rows; ++i) {
    for (size_t j = 0; j < y_true.cols; ++j) {
      VALUE_AT(cost_t, i, j) =
          pow((VALUE_AT(y_true, i, j) - VALUE_AT(y_pred, i, j)), 2);
    }
  }
  return cost_t;
}
Tensor new_target(Tensor activation, Tensor target, Tensor pred, Tensor weight,
                  ActivationFunction act_fun) {
  Tensor t = new_tensor(activation.rows, activation.cols);
  for (size_t i = 0; i < activation.rows; ++i) {
    for (size_t j = 0; j < activation.cols; ++j) {
      if (act_fun == sigmoid)
        VALUE_AT(t, i, j) = (VALUE_AT(target, i, j) - VALUE_AT(pred, i, j)) *
                            VALUE_AT(pred, i, j) * (1 - VALUE_AT(pred, i, j)) *
                            VALUE_AT(weight, i, j);
      if (act_fun == relu) {
        float d_relu_act = VALUE_AT(pred, i, j) >= 0 ? 1 : 0;
        VALUE_AT(t, i, j) = (VALUE_AT(target, i, j) - VALUE_AT(pred, i, j)) *
                            d_relu_act * VALUE_AT(weight, i, j);
      }
    }
  }
  return t;
}

void _backward(Net nn, Tensor target) {
  // start from the last layer
  //  act function of last layer is relu
  // mse = 1/n * sum((y - yp)^2)
  // y = sigmoid(last_layer_activations)
  // dcost/doutput_layer =
  // loop through each layer
  for (size_t i = nn.net_size; i > 0; --i) {
    // for the output layerxw
    if (i == nn.net_size) {
      d_sigmoid(nn.activations[i], nn.activations[i - 1],
                nn.grad_weights[i - 1], target, 1);

      d_sigmoid(nn.activations[i], nn.activations[i - 1], nn.grad_biases[i - 1],
                target, 0);
      // update the weights
      _update(nn.weights[i - 1], nn.grad_weights[i - 1]);
      // update the biases
      _update(nn.biases[i - 1], nn.grad_biases[i - 1]);
      continue;
    }
    target = new_target(nn.activations[i], target, nn.activations[i + 1],
                        nn.weights[i + 1], nn.activation_funs[i + 1]);

    // all other layers are considered to have relu activations
    // update weights
    d_relu(nn.activations[i], nn.activations[i - 1], nn.grad_weights[i - 1],
           target, 1);
    PRINT_T(nn.weights[i - 1]);
    _update(nn.weights[i - 1], nn.grad_weights[i - 1]);
    PRINT_T(nn.weights[i - 1]);
    // update biases
    d_relu(nn.activations[i], nn.activations[i - 1], nn.grad_biases[i - 1],
           target, 0);
    _update(nn.biases[i - 1], nn.grad_biases[i - 1]);

    free(target.es);
  }
}

void _update(Tensor params, Tensor grad) {
  assert(params.rows == grad.rows);
  assert(params.cols == grad.cols);

  for (size_t i = 0; i < params.rows; ++i) {
    for (size_t j = 0; j < params.cols; ++j) {
      VALUE_AT(params, i, j) -= VALUE_AT(grad, i, j);
    }
  }
}

void free_neural_net(Net *nn) {
  free(nn->activations[nn->net_size].es);
  for (size_t i = 0; i < nn->net_size; ++i) {
    free(nn->weights[i].es);
    free(nn->biases[i].es);
    free(nn->grad_weights[i].es);
    free(nn->grad_biases[i].es);
    free(nn->activations[i].es);
  }
  free(nn->weights);
  free(nn->biases);
  free(nn->grad_weights);
  free(nn->grad_biases);
  free(nn->activations);
}

void print_tensor(Tensor m, char *name) {
  printf("%s = [\n", name);
  for (size_t i = 0; i < m.rows; ++i) {
    printf("\t[");
    for (size_t j = 0; j < m.cols; j++) {
      printf("%f,", VALUE_AT(m, i, j));
    }
    printf("],\n");
  }
  printf("]\n");
  return;
}

void print_network(Net nn, char *name) {
  printf("Network Summary for network %s.... \n", name);
  char buf[256];
  print_tensor(nn.activations[0], "Input");
  for (size_t i = 0; i < nn.net_size; ++i) {
    sprintf(buf, "W%lu", i);
    print_tensor(nn.weights[i], buf);

    sprintf(buf, "B%lu", i);
    print_tensor(nn.biases[i], buf);

    sprintf(buf, "A%lu", i + 1);
    print_tensor(nn.activations[i + 1], buf);
  }
}
#endif // NN_IMPLEMENTATION
