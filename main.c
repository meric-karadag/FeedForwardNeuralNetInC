#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix/matrix.h"
#include "matrix/matrixOps.h"
#include "image/img.h"
#include "ffNN/activations.h"
#include "ffNN/nn.h"

int main()
{
  srand(time(NULL));

  // Training
  int numImages = 10000;
  int epochs = 5;
  Img **imgs = csvToImg("./data/mnist_train.csv", numImages, 28, 28);
  NeuralNet *nn = networkCreate(784, 300, 10, 0.001);
  networkTrainBatchImgs(nn, imgs, numImages, epochs);

  return 0;
}