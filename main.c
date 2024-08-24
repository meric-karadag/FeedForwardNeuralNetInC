#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

#include "matrix/matrix.h"
#include "matrix/matrixOps.h"
#include "image/img.h"
#include "ffNN/activations.h"
#include "ffNN/nn.h"

int getArg(int argc, char *argv[], int index, int defaultValue)
{
  if (index < argc)
  {
    return atoi(argv[index]);
  }
  return defaultValue;
}

double getDoubleArg(int argc, char *argv[], int index, double defaultValue)
{
  if (index < argc)
  {
    return atof(argv[index]);
  }
  return defaultValue;
}

int fileExists(const char *filePath)
{
  FILE *file = fopen(filePath, "r");
  if (file)
  {
    fclose(file);
    return 1; // File exists
  }
  return 0; // File does not exist or is not accessible
}

int main(int argc, char *argv[])
{
  // Default parameter values
  int epochs = 5;
  int hiddenDim = 300;
  double learningRate = 1e-3;
  int numTrainImages = 10000;
  int numTestImages = 1000;
  int imgDimRow = 28;
  int imgDimCol = 28;
  int numClasses = 10;
  char *trainingDataPath = NULL;
  char *testDataPath = NULL;
  char *savePath = NULL;
  char *logFilePath = NULL;

  // Command-line options parsing with getopt
  int opt;
  while ((opt = getopt(argc, argv, "t:T:e:l:h:d:r:s:R:S:c:o:")) != -1)
  {
    switch (opt)
    {
    case 't':
      trainingDataPath = optarg;
      break;
    case 'T':
      testDataPath = optarg;
      break;
    case 'e':
      epochs = atoi(optarg);
      break;
    case 'l':
      learningRate = atof(optarg);
      break;
    case 'h':
      hiddenDim = atoi(optarg);
      break;
    case 'd':
      savePath = optarg;
      break;
    case 'r':
      numTrainImages = atoi(optarg);
      break;
    case 's':
      numTestImages = atoi(optarg);
      break;
    case 'R':
      imgDimRow = atoi(optarg);
      break;
    case 'S':
      imgDimCol = atoi(optarg);
      break;
    case 'c':
      numClasses = atoi(optarg);
      break;
    case 'o':
      logFilePath = optarg;
      break;
    default:
      fprintf(stderr, "Usage: %s -t <training_data.csv> -T <test_data.csv> -e <epochs> -l <learning_rate> -h <hidden_dim> -d <save_path> -r <num_train_images> -s <num_test_images> -R <img_dim_row> -S <img_dim_col> -c <num_classes> -o <log_file_path>\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if (!trainingDataPath || !testDataPath || !savePath)
  {
    fprintf(stderr, "Usage: %s -t <training_data.csv> -T <test_data.csv> -e <epochs> -l <learning_rate> -h <hidden_dim> -d <save_path> -r <num_train_images> -s <num_test_images> -R <img_dim_row> -S <img_dim_col> -c <num_classes> -o <log_file_path>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Check if the training and test data files exist
  if (!fileExists(trainingDataPath))
  {
    fprintf(stderr, "Error: Training data file '%s' does not exist or cannot be opened.\n", trainingDataPath);
    exit(EXIT_FAILURE);
  }

  if (!fileExists(testDataPath))
  {
    fprintf(stderr, "Error: Test data file '%s' does not exist or cannot be opened.\n", testDataPath);
    exit(EXIT_FAILURE);
  }

  // Open log file if specified
  FILE *logFile = NULL;
  if (logFilePath)
  {
    logFile = fopen(logFilePath, "w");
    if (!logFile)
    {
      fprintf(stderr, "Error: Could not open log file '%s'.\n", logFilePath);
      exit(EXIT_FAILURE);
    }
  }

  printf("Training with parameters:\n");
  printf("Training Data: %s\n", trainingDataPath);
  printf("Test Data: %s\n", testDataPath);
  printf("Save Path: %s\n", savePath);
  printf("Epochs: %d\n", epochs);
  printf("Hidden Layer Size: %d\n", hiddenDim);
  printf("Learning Rate: %f\n", learningRate);
  printf("Number of Train Images: %d\n", numTrainImages);
  printf("Number of Test Images: %d\n", numTestImages);
  printf("Image Dimensions: %dx%d\n", imgDimRow, imgDimCol);
  printf("Number of Classes: %d\n", numClasses);

  srand(time(NULL));

  // Load training and test data
  Img **trainImgs = csvToImg(trainingDataPath, numTrainImages, imgDimRow, imgDimCol);
  Img **testImgs = csvToImg(testDataPath, numTestImages, imgDimRow, imgDimCol);

  // Training
  NeuralNet *nn = networkCreate(imgDimRow * imgDimCol, hiddenDim, numClasses, learningRate);
  networkTrain(nn, trainImgs, numTrainImages, testImgs, numTestImages, epochs, savePath, logFile);

  // Cleanup
  networkFree(nn);
  imgsFree(trainImgs, numTrainImages);
  imgsFree(testImgs, numTestImages);
}