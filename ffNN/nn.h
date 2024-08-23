#ifndef NN
#define NN

#include "../matrix/matrix.h"
#include "../image/img.h"

typedef struct
{
  int inputDim;
  int hiddenDim;
  int outputDim;
  double learningRate;
  Matrix *hiddenLayer;
  Matrix *outputLayer;
} NeuralNet;

NeuralNet *networkCreate(int inputDim, int hiddenDim, int outputDim, double learningRate);
double networkStep(NeuralNet *nn, Matrix *inputs, Matrix *outputs);
void networkTrainBatchImgs(NeuralNet *nn, Img **imgs, int batchSize, int epochs);
Matrix *networkPredictImg(NeuralNet *nn, Img *img);
double networkPredictImgs(NeuralNet *nn, Img **imgs, int n);
Matrix *networkPredict(NeuralNet *nn, Matrix *input);
void networkSave(NeuralNet *nn, char *fileName);
NeuralNet *networkLoad(char *fileName);
void networkPrint(NeuralNet *nn);
void networkFree(NeuralNet *nn);

#endif