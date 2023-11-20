
#ifndef NN_H_
#define NN_H_

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define VALUE_AT(t, i, j) (t).es[(i) * (t).cols + (j)]

typedef struct {
  /*define a matrix
   * with arbitary size
   */
  size_t rows;
  size_t cols;
  float *es; // pointer to the memory that hold the float(values)
} Tensor;

typedef Tensor (*ActivationFunction)(Tensor);

float float_rand(void);
float _sigmoid(float);

Tensor new_tensor(size_t, size_t);
// Tensor initializations
void rand_tensor(Tensor, int, int);
void zeros_tensor(Tensor);
void ones_tensor(Tensor);
void set_tensor(Tensor, int rows,int cols,float (*)[cols]);
// Tensor operation
Tensor matmul(Tensor, Tensor);
Tensor matadd(Tensor, Tensor);
Tensor sigmoid(Tensor);
Tensor relu(Tensor);
// Basic nn
Tensor *linear_layer(int, int, int, int *);
Tensor feed_forward(int, Tensor, Tensor *, ActivationFunction *);
void print_tensor(Tensor, char *);

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
      VALUE_AT(m, i, j) = 0.0;
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

void set_tensor(Tensor m,int rows, int cols,float (*values)[cols]){
  for(int i = 0; i < rows; ++i){
    for(int j = 0; j < cols; j++){
      VALUE_AT(m,i,j) = values[i][j];
    }
  }
}

Tensor matmul(Tensor a, Tensor b) {
  /* Tensor multiplication between to matrices
   * Args:
   *  a(Tensor): firs matrice
   *  b(Tensor): second matrice
   * a.cols == b.rows
   * Returns:
   *  Tensor
   * */
  assert(a.cols == b.rows);
  // if compatible create a new matrice with
  // shape a.rows x b.cols
  Tensor dst = new_tensor(a.rows, b.cols);

  // multiply ith row of matrice a with jth col of matrice b
  for (size_t i = 0; i < a.rows; ++i) {
    for (size_t j = 0; j < b.cols; ++j) {
      VALUE_AT(dst, i, j) = 0;
      for (size_t k = 0; k < a.rows; ++k) {
        VALUE_AT(dst, i, j) += VALUE_AT(a, i, k) * VALUE_AT(b, k, j);
      }
    }
  }
  return dst;
}

Tensor matadd(Tensor a, Tensor b) {
  /* Elementwise summation of two matrices
   * matrices should have the same dimension
   * Args:
   *  dst(Tensor): destination matrice
   *  src(Tensor): source matrice
   * */
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);

  Tensor dst = new_tensor(a.rows, b.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      VALUE_AT(dst, i, j) = VALUE_AT(a, i, j) + VALUE_AT(b, i, j);
    }
  }
  return dst;
}

/*Operations*/

float _sigmoid(float x) { return 1 / (1 + expf(-x)); }

Tensor sigmoid(Tensor x) {
  Tensor dst = new_tensor(x.rows, x.cols);

  for (size_t i = 0; i < x.rows; ++i) {
    for (size_t j = 0; j < x.cols; ++j) {
      VALUE_AT(dst, i, j) = _sigmoid(VALUE_AT(x, i, j));
    }
  }
  return dst;
}

Tensor relu(Tensor x) {
  Tensor dst = new_tensor(x.rows, x.cols);
  for (size_t i = 0; i < x.rows; ++i) {
    for (size_t j = 0; j < x.cols; ++j) {
      VALUE_AT(dst, i, j) = fmaxf(0, VALUE_AT(x, i, j));
    }
  }
  return dst;
}

Tensor *linear_layer(int input_size, int output_size, int num_h_layers,
                     int *h_layer_sizes) {
  // The input is assumed to be a column vector (1, input_size)
  //  weights (input_size,output_size)

  //  input dot weights
  Tensor *weights =
      calloc(num_h_layers + 1, sizeof(Tensor)); // one for output layer
  int rows = input_size;
  int cols = *h_layer_sizes;

  Tensor in_weight = new_tensor(rows, cols);
  rand_tensor(in_weight, -1, 1);

  weights[0] = in_weight;
  // hidden layers
  for (int i = 1; i <= num_h_layers - 1; ++i) {
    Tensor h_weight = new_tensor(cols, h_layer_sizes[i]);
    rand_tensor(h_weight, -1, 1);
    weights[i] = h_weight;
    cols = h_layer_sizes[i];
  }
  // output layer
  Tensor out_weight = new_tensor(cols, output_size);
  rand_tensor(out_weight, -1, 1);
  weights[num_h_layers] = out_weight;

  return weights;
}

Tensor feed_forward(int layer_size, Tensor input, Tensor *weights,
                    ActivationFunction *activations) {
  // Perform matmul between input and weight
  // add bies (ignored for simplicity)
  // activation function
  Tensor logits = activations[0](matmul(input, *weights));

  for (int i = 1; i < layer_size; ++i) {
    logits = activations[i](matmul(logits, weights[i]));
  }
  return logits;
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
#endif // NN_IMPLEMENTATION
