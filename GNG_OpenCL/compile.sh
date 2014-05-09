#!/bin/bash
g++ -I /opt/AMDAPP/include -L /opt/AMDAPP/lib/x86_64 -o OPENCL_NN  fullyconnectedneuralnet.cpp convolutionalneuralnet.cpp neuralnet.cpp distortions.cpp training.cpp layer.cpp MNIST.cpp main.cpp -lOpenCL -I. -std=c++11

#-msse2 -m64
#-I/opt/AMDAPP/samples/opencl/cl/HelloWorld/../../../../include/SDKUtil -I/include/SDKUtil   


