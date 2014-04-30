#ifndef OPENCL_GNG_NETWORK_H
#define OPENCL_GNG_NETWORK_H


struct NeuralGasNetwork;
struct Node;
struct NodeEdge;
struct IntStack;

typedef struct IntStack{
    int headIndex;
    int stack[MAX_STACK_SIZE];
} IntStack;

//Pre: stackSize < MAX_STACK_SIZE
//Creates and returns an IntStack with the values 0,1, 2, 3, ..., (stacksize-1)

bool IntStack_isEmpty(global IntStack* intStack)
{
    if (intStack->headIndex == -1)
        return true;
    return false;
}

int IntStack_pop(global IntStack* intStack)
{
    --intStack->headIndex;
    return intStack->stack[intStack->headIndex + 1];
}

void IntStack_push(global IntStack* intStack, int n)
{
    ++intStack->headIndex;
    intStack->stack[intStack->headIndex] = n;
}
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