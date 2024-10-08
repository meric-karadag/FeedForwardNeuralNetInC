#include "nn.h"
#include "activations.h"
#include "../matrix/matrixOps.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#define MAXCHAR 1000

NeuralNet *networkCreate(int inputDim, int hiddenDim, int outputDim, double learningRate)
{
  NeuralNet *nn = malloc(sizeof(NeuralNet));
  nn->inputDim = inputDim;
  nn->hiddenDim = hiddenDim;
  nn->outputDim = outputDim;
  nn->learningRate = learningRate;

  Matrix *hiddenLayer = matrixCreate(inputDim, hiddenDim);
  Matrix *outputLayer = matrixCreate(hiddenDim, outputDim);

  matrixRandomInit(hiddenLayer, 0, 0.3);
  matrixRandomInit(outputLayer, 0, 0.3);

  nn->hiddenLayer = hiddenLayer;
  nn->outputLayer = outputLayer;

  return nn;
}

double networkStep(NeuralNet *nn, Matrix *inputs, Matrix *outputs, int *predictedLabel)
{
  // Forward pass

  // (n, hiddenDim) = (n, inputDim) @ (inputDim, hiddenDim)
  Matrix *h1 = dot(inputs, nn->hiddenLayer);

  Matrix *z1 = apply(h1, relu);

  // (n, outDim) = (n, hiddenDim) @ (hiddenDim, outDim)
  Matrix *h2 = dot(z1, nn->outputLayer);

  Matrix *z2 = softmax(h2);
  *predictedLabel = matrixArgmax(z2);

  // printf("Predicted Label: %d\n", *predictedLabel);
  // matrixPrint(z2);

  // Calculate loss
  double loss = crossEntropyLoss(z2, outputs);

  // Backpropagation

  // Get necessary transposed matrices for backprop
  Matrix *outputLayerT = transpose(nn->outputLayer);
  Matrix *z1T = transpose(z1);
  Matrix *inputsT = transpose(inputs);
  Matrix *z1Prime = apply(z1, reluPrime);
  // Calculate error and take partial derivates

  // (n, outDim) = (n, outDim) - (n, outDim)
  Matrix *outputError = elementWiseSubtract(z2, outputs);

  // d Loss / d OutputLayer
  // (hiddenDim, outputDim) = (hiddenDim, n) @ (n, outDim)
  Matrix *outputGradient = dot(z1T, outputError);

  // d Loss / d z1
  // (n, hiddenDim) = (n, outDim) @ (outDim, hiddenDim)
  Matrix *hiddenError = dot(outputError, outputLayerT);

  // d Loss / d h1
  // (n, hiddenDim) = (n, hiddenDim) * (n, hiddenDim)
  Matrix *hiddenGradient = elementWiseMultiply(hiddenError, z1Prime);

  // d Loss / d HiddenLayer
  // (inputDim, hiddenDim) = (inputDim, n) @ (n, hiddenDim)
  Matrix *hiddenLayerGradient = dot(inputsT, hiddenGradient);

  // Update weights
  Matrix *scaledOutputLayerGradients = scale(outputGradient, nn->learningRate);
  Matrix *newOutputLayer = elementWiseSubtract(nn->outputLayer, scaledOutputLayerGradients);
  matrixFree(nn->outputLayer);
  nn->outputLayer = newOutputLayer;

  Matrix *scaledHiddenLayerGradients = scale(hiddenLayerGradient, nn->learningRate);
  Matrix *newHiddenLayer = elementWiseSubtract(nn->hiddenLayer, scaledHiddenLayerGradients);
  matrixFree(nn->hiddenLayer);
  nn->hiddenLayer = newHiddenLayer;

  // Free matrices
  matrixFree(h1);
  matrixFree(z1);
  matrixFree(h2);
  matrixFree(z2);
  matrixFree(outputError);
  matrixFree(hiddenError);
  matrixFree(hiddenGradient);
  matrixFree(outputGradient);
  matrixFree(hiddenLayerGradient);
  matrixFree(outputLayerT);
  matrixFree(z1T);
  matrixFree(inputsT);
  matrixFree(scaledOutputLayerGradients);
  matrixFree(scaledHiddenLayerGradients);
  matrixFree(z1Prime);
  return loss;
}

// Function to log progress to a file
void logProgress(FILE *logFile, const char *format, ...)
{
  if (logFile)
  {
    va_list args;
    va_start(args, format);
    vfprintf(logFile, format, args);
    va_end(args);
  }
}

void networkTrain(NeuralNet *nn, Img **trainImgs, int numTrainImgs, Img **testImgs, int numTestImgs, int epochs, char *savePath, FILE *logFile)
{
  int i, j, predLabel, numCorrect = 0;
  double bestAccuracy = 0;

  for (j = 1; j <= epochs; j++)
  {
    numCorrect = 0;
    double loss = 0;
    printf("Training >> epoch:%d/%d\n", j, epochs);
    logProgress(logFile, "Training >> epoch:%d/%d\n", j, epochs);

    for (i = 0; i < numTrainImgs; i++)
    {
      if (i % 100 == 0 && i != 0)
      {
        printf("Iter No. %d, Train Loss : %f, Train Accuracy : %f\n", i, loss / (i + 1), (double)numCorrect / i);
        logProgress(logFile, "Iter No. %d, Train Loss : %f, Train Accuracy : %f\n", i, loss / (i + 1), (double)numCorrect / i);
      }

      Img *curImg = trainImgs[i];
      Matrix *input = matrixFlatten(curImg->imgData, 1); // 0 -> (1, x) row vector
      Matrix *output = matrixCreate(1, nn->outputDim);   // (1, outDim) one hot vector
      matrixFill(output, 0);
      output->entries[0][curImg->label] = 1; // Set the correct label in one hot vector
      loss += networkStep(nn, input, output, &predLabel);
      if (predLabel == curImg->label)
      {
        numCorrect++;
      }
      matrixFree(input);
      matrixFree(output);
    }

    printf("Epoch %d, Train Loss : %f, Train Accuracy : %f\n", j, loss / numTrainImgs, (double)numCorrect / numTrainImgs);
    logProgress(logFile, "Epoch %d, Train Loss : %f, Train Accuracy : %f\n", j, loss / numTrainImgs, (double)numCorrect / numTrainImgs);

    double testAccuracy = networkPredictImgs(nn, testImgs, numTestImgs);
    printf("Epoch %d, Test Accuracy : %f\n", j, testAccuracy);
    logProgress(logFile, "Epoch %d, Test Accuracy : %f\n", j, testAccuracy);

    if (testAccuracy > bestAccuracy)
    {
      printf("Test accuracy improved from %f to %f. Saving model...\n", bestAccuracy, testAccuracy);
      logProgress(logFile, "Test accuracy improved from %f to %f. Saving model...\n", bestAccuracy, testAccuracy);
      bestAccuracy = testAccuracy;
      networkSave(nn, savePath);
    }
  }
}

Matrix *networkPredictImg(NeuralNet *nn, Img *img)
{
  Matrix *input = matrixFlatten(img->imgData, 1);
  Matrix *result = networkPredict(nn, input);
  matrixFree(input);
  return result;
}

double networkPredictImgs(NeuralNet *nn, Img **imgs, int n)
{
  int i, numCorrect = 0;
  for (i = 0; i < n; i++)
  {
    Matrix *pred = networkPredictImg(nn, imgs[i]);
    if (matrixArgmax(pred) == imgs[i]->label)
    {
      numCorrect++;
    }
    matrixFree(pred);
  }

  return (double)numCorrect / n;
}

Matrix *networkPredict(NeuralNet *nn, Matrix *input)
{
  Matrix *h1 = dot(input, nn->hiddenLayer);
  Matrix *z1 = apply(h1, relu);
  Matrix *h2 = dot(z1, nn->outputLayer);
  Matrix *z2 = softmax(h2);

  matrixFree(h1);
  matrixFree(z1);
  matrixFree(h2);

  return z2;
}

void networkSave(NeuralNet *nn, const char *directoryName)
{
  // Create the directory if it doesn't exist
  mkdir(directoryName, 0777);

  // Build file paths
  char descriptionPath[256];
  char hiddenLayerPath[256];
  char outputLayerPath[256];

  snprintf(descriptionPath, sizeof(descriptionPath), "%s/description", directoryName);
  snprintf(hiddenLayerPath, sizeof(hiddenLayerPath), "%s/hidden", directoryName);
  snprintf(outputLayerPath, sizeof(outputLayerPath), "%s/output", directoryName);

  // Save description
  FILE *description = fopen(descriptionPath, "w");
  if (!description)
  {
    fprintf(stderr, "Error: Could not open file %s for writing.\n", descriptionPath);
    return;
  }
  fprintf(description, "%d\n", nn->inputDim);
  fprintf(description, "%d\n", nn->hiddenDim);
  fprintf(description, "%d\n", nn->outputDim);
  fclose(description);

  // Save matrices
  matrixSave(nn->hiddenLayer, hiddenLayerPath);
  matrixSave(nn->outputLayer, outputLayerPath);

  printf("Successfully written network to directory '%s'\n", directoryName);
}

NeuralNet *networkLoad(char *fileName)
{
  NeuralNet *nn = malloc(sizeof(NeuralNet));
  char entry[MAXCHAR];
  chdir(fileName);

  FILE *description = fopen("description", "r");
  fgets(entry, MAXCHAR, description);
  nn->inputDim = atoi(entry);

  fgets(entry, MAXCHAR, description);
  nn->hiddenDim = atoi(entry);

  fgets(entry, MAXCHAR, description);
  nn->outputDim = atoi(entry);
  fclose(description);

  nn->hiddenLayer = matrixLoad("hidden");
  nn->outputLayer = matrixLoad("output");

  printf("Succesfully loaded neural network from %s\n", fileName);
  return nn;
}

void networkPrint(NeuralNet *nn)
{
  printf("# of Inputs: %d\n", nn->inputDim);
  printf("# of Hidden: %d\n", nn->hiddenDim);
  printf("# of Output: %d\n", nn->outputDim);
  printf("Hidden Weights: \n");
  matrixPrint(nn->hiddenLayer);
  printf("Output Weights: \n");
  matrixPrint(nn->outputLayer);
}

void networkFree(NeuralNet *nn)
{
  matrixFree(nn->hiddenLayer);
  matrixFree(nn->outputLayer);
  free(nn);
  nn = NULL;
}
