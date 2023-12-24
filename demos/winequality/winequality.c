#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NN_IMPLEMENTATION
#include "../../nn.h"

#define MAX_LINE_SIZE 2046 
#define MAX_FIELD_SIZE 256
#define NUM_SAMPLES 4898
#define NUM_INPUTS 11
#define NUM_OUTPUTS 1
#define CSV_COL_SIZE (NUM_INPUTS + NUM_OUTPUTS )

FILE *open_csv_file(char *filename)
{
  FILE* file = fopen(filename, "r");
  if (!file)
  {
    // Handle error: print message, exit program etc.
    fprintf(stderr, "Error reading file ..\n");
  }
  return file;
}

void validate(Tensor X, Tensor Y, Net model, size_t batch_size, size_t train_size)
{
  printf("==========================\n");
  printf("Starting Evaluation ... \n");
  float epoch_loss = 0.;
  size_t true_label_count = 0;
  size_t count = 0;
  size_t test_size = (X.rows - train_size);
  size_t steps = (unsigned long long)test_size / batch_size;

  printf("Steps = %lu\n", steps);
  for (size_t step = 0; step < steps; ++step)
  {
    Tensor *batch = get_batch(X, Y, train_size, NUM_INPUTS, NUM_OUTPUTS, batch_size);
    Tensor x_batch = batch[0];
    Tensor y_batch = batch[1];

    for (size_t i = 0; i < x_batch.rows; ++i)
    {
      Tensor x_test = get_row(x_batch, i, x_batch.cols);
      Tensor y_test = get_row(y_batch, i, y_batch.cols);
      // normalize input
      normalize_tensor(x_test);
      // forward
      Tensor pred = _forward(model, x_test);
      // threshold
      Tensor mse_cost = mse(y_test, pred);
      epoch_loss += *mse_cost.es;

      threshold(pred, 0.5);
      //printf("Y_true = %f, Pred = %f\n", *y_test.es, *pred.es);
      if (*y_test.es == *pred.es)
      {
        true_label_count++;
      }
      free(x_test.es);
      free(y_test.es);
      free(mse_cost.es);
    }
    epoch_loss /= batch_size;
    free(x_batch.es);
    free(y_batch.es);
    free(batch);
    count++;
  }
  epoch_loss /= count;
  float test_acc = (float)true_label_count / test_size;

  printf("Count =  %lu\n", count);
  printf("True label count =  %lu\n", true_label_count);
  printf("Validation loss = %f\n", epoch_loss);
  printf("Validation Accuracy = %f\n", test_acc * 100);
}
void train(Tensor X, Tensor Y, Net model, float lr, size_t batch_size, size_t epochs, size_t train_size)
{
  size_t steps = (unsigned long long)train_size / batch_size;

  printf("Training size = %lu\n", train_size);
  printf("Batch size = %lu\n", batch_size);
  printf("Step size = %lu\n", steps);

  for (size_t epoch = 0; epoch < epochs; ++epoch)
  {
    printf("*******************************************\n");
    printf("Training for Epoch %lu\n", epoch);
    printf("*******************************************\n");
    // get batch
    float epoch_loss = 0.;
    size_t true_label_count = 0;
    for (size_t step = 0; step < steps; ++step)
    {
      Tensor *batch = get_batch(X, Y, 0, NUM_INPUTS, NUM_OUTPUTS, batch_size);
      Tensor x_batch = batch[0];
      Tensor y_batch = batch[1];
      for (size_t i = 0; i < x_batch.rows; ++i)
      {
        Tensor x_train = get_row(x_batch, i, x_batch.cols);
        Tensor y_train = get_row(y_batch, i, y_batch.cols);
        // normalize input
        normalize_tensor(x_train);
        // forward
        Tensor pred = _forward(model, x_train);

        Tensor mse_cost = mse(y_train, pred);
        epoch_loss += *mse_cost.es;
        // backward
        _backward(model, y_train, pred);

        threshold(pred, 0.5);
       // printf("Y_true = %f, Pred = %f\n", *y_train.es, *pred.es);

        if (*y_train.es == *pred.es)
        {
          true_label_count++;
        }

        free(x_train.es);
        free(y_train.es);
        free(mse_cost.es);
      }
      epoch_loss /= batch_size;
      _update(model, lr, batch_size);
      zero_grad(model);
      free(x_batch.es);
      free(y_batch.es);
      free(batch);
    }
    epoch_loss /= steps;
    float train_acc = (float)true_label_count / train_size;
    printf("loss for epoch %lu = %f\n", epoch + 1, epoch_loss);
    printf("Accuracy for epoch %lu= %f\n", epoch + 1, train_acc * 100);
  }
    validate(X, Y, model, batch_size, train_size);
}

int main(void)
{
  srand(time(0));
  // Open the csv file
  Tensor X = new_tensor(NUM_SAMPLES, NUM_INPUTS);
  Tensor Y = new_tensor(NUM_SAMPLES, NUM_OUTPUTS);
  char file_path[] = "../../datasets/winequality.csv";

  FILE* file = open_csv_file(file_path);
  char* line = (char*) calloc(MAX_LINE_SIZE,sizeof(char));
  int row = 0;
  printf("Reading CSV data and parsing to a Tensor\n");
  //while ((line = read_line(file)) != NULL)
  while(fgets(line,MAX_LINE_SIZE,file) != NULL && row < NUM_SAMPLES)
  {
    if (row != 0){
      char* token;
      // get the first token
      token = strtok(line,";");
      int i = 0;

      while (token != NULL)
      {
        //printf("Token %d = %s\n",i,token);
        if (i < 11)
          VALUE_AT(X, row, i) = strtof(token, NULL);
        else
          VALUE_AT(Y, row, 0) = strtof(token, NULL) >= 6 ? 1. : 0.;
        i++;
        token = strtok(NULL,";");
      }
    }
    row++;
  }
  fclose(file);
  free(line);   // Free memory allocated for line
  //free(tokens);
  //
  // create the neural net
  size_t arch[] = {NUM_INPUTS, 32, 16, 8, 4, 2, 1};
  Activation activations[] = {RELU, RELU, RELU, RELU, RELU, SIGMOID};
  size_t count = ARRAY_SIZE(arch);
  Net model = fully_connected_layer(arch, count, activations);

  float train_ratio = 0.8;
  size_t train_size = (unsigned long long)(X.rows * train_ratio);

  train(X, Y, model, 1e-2, 16, 1, train_size);
  //PRINT_T(Y);
  printf("Finished Training, freeing up used memory ... \n");
  free(X.es);
  free(Y.es);
  free_neural_net(&model);
  return 0;
}
