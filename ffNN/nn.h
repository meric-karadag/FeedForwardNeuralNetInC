#ifndef NN
#define NN

#include "../matrix/matrix.h"
#include "../image/img.h"
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

typedef struct
{
  int inputDim;
  int hiddenDim;
  int outputDim;
  double learningRate;
  Matrix *hiddenLayer;
  Matrix *outputLayer;
} NeuralNet;

void logProgress(FILE *logFile, const char *format, ...);
NeuralNet *networkCreate(int inputDim, int hiddenDim, int outputDim, double learningRate);
double networkStep(NeuralNet *nn, Matrix *inputs, Matrix *outputs, int *predictedLabel);
void networkTrain(NeuralNet *nn, Img **trainImgs, int numTrainImgs, Img **testImgs, int numTestImgs, int epochs, char *savePath, FILE *logFile);
Matrix *networkPredictImg(NeuralNet *nn, Img *img);
double networkPredictImgs(NeuralNet *nn, Img **imgs, int n);
Matrix *networkPredict(NeuralNet *nn, Matrix *input);
void networkSave(NeuralNet *nn, const char *fileName);
NeuralNet *networkLoad(char *fileName);
void networkPrint(NeuralNet *nn);
void networkFree(NeuralNet *nn);

#endif