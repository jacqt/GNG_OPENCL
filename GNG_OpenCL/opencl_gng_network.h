#ifndef OPENCL_GNG_NETWORK_H
#define OPENCL_GNG_NETWORK_H

#include "opencl_utils.cl"

#define NODE_CHANGE_RATE                0.05
#define NODE_NEIGHBOR_CHANGE_RATE       0.00005
#define LOCAL_DECREASE_RATE             0.5
#define GLOBAL_DECREASE_RATE            0.9995
#define AGE_MAX                         3200
#define TIME_BETWEEN_ADDING_NODES       300
#define MAX_NODES                       1000
#define MAX_EDGES                       1000
#define MAX_DIMENSION                   200
#define MAX_NODES                       1000
#define MAX_EDGES                       1000
#define MAX_DIMENSION                   200

#define CL_TRUE                         42

struct NeuralGasNetwork;
struct Node;
struct NodeEdge;

typedef struct NodeEdge{
    bool isNull;
    unsigned int age;
    unsigned int nodeIndex1;
    unsigned int nodeIndex2;
} NodeEdge;

typedef struct Node{
    bool isNull;
    unsigned int featureDimension;
    float referenceVector[MAX_DIMENSION];
	//NodeEdge edges[MAX_EDGES];
    float error;
    float distance;
} Node;

typedef struct NeuralGasNetwork {
    unsigned int numberOfClassifications;
    Node nodes[MAX_NODES];
	NodeEdge edges[MAX_EDGES];
    IntStack emptyNodeIndexStack;
    IntStack emptyEdgeIndexStack;
} NeuralGasNetwork;


void shiftReferenceVector(global Node* node, global float* inputVector, float changeRate)
{
    //
}

NodeEdge NodeEdge_createEdge(
    NeuralGasNetwork* network,
	unsigned int n1,
	unsigned int n2,
	unsigned int age_); //Looks for the first non empty spot

void NodeEdge_deleteEdge();

#endif