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

  int imgDimRow = 28;
  int imgDimCol = 28;

  int numImages = 10000;
  Img **imgs = csvToImg("./data/mnist_train.csv", numImages, imgDimRow, imgDimCol);

  int numTestImages = 1000;
  Img **testImgs = csvToImg("./data/mnist_test.csv", numTestImages, imgDimRow, imgDimCol);

  char fileName[] = "mnist_nn";
  // Training
  int epochs = 5;
  int hiddenDim = 300;
  int numClasses = 10;
  double learningRate = 1e-3;
  NeuralNet *nn = networkCreate(imgDimRow * imgDimCol, hiddenDim, numClasses, learningRate);
  networkTrainBatchImgs(nn, imgs, numImages, epochs);
  networkSave(nn, fileName);
  networkFree(nn);
  imgsFree(imgs, numImages);

  // Testing
  NeuralNet *nnLoaded = networkLoad(fileName);
  double accuracy = networkPredictImgs(nnLoaded, testImgs, numTestImages);
  printf("Test Accuracy: %f\n", accuracy);
  imgsFree(testImgs, numTestImages);
  networkFree(nnLoaded);

  return 0;
}