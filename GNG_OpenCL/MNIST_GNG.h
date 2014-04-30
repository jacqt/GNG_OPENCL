#ifndef MNIST_GNG_H
#define MNIST_GNG_H

#include "general_include.h"
#include "gpu_gng.h"
#include "cpu_gng.h"
#include "graph_algorithms.h"
#include "convert_cpu_gpu_gngs.h"

void readMNIST(vector<double*> &inputs, vector<int*> &targets, int &inputSize);
void readMNISTTest(vector<double*> &inputs, vector<int*> &targets, int &inputSize);

void trainMNIST_GNG();

void testMNIST_GNG();

void yolo();
//void calcMeanSquaredError(vector<double*> &inputs, NeuralGasNetwork* network);

#endif
