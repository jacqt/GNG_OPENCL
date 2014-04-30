#define NODE_CHANGE_RATE                0.03
#define NODE_NEIGHBOR_CHANGE_RATE       0.0003
#define LOCAL_DECREASE_RATE             0.5
#define GLOBAL_DECREASE_RATE            0.9995
#define AGE_MAX                         15000
#define TIME_BETWEEN_ADDING_NODES       300
#define MAX_NODES                       1024
#define MAX_EDGES                       2048
#define MAX_DIMENSION                   1024

#define MAX_STACK_SIZE                  2100
#define OPENCL_TRUE                     -124


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
    //printf("WTF 2.0? : %d\n", intStack->headIndex);
    --intStack->headIndex;
    return intStack->stack[intStack->headIndex];
}

void IntStack_push(global IntStack* intStack, int n)
{
    //printf("WTF? : %d\n", intStack->headIndex);
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
    Node nodes[MAX_NODES]; NodeEdge edges[MAX_EDGES];
    IntStack emptyNodeIndexStack;
    IntStack emptyEdgeIndexStack;
} NeuralGasNetwork;


void shiftReferenceVector(global Node* node, global float* inputVector, float changeRate)
{
    //printf("???: %f, %f\n", node->referenceVector[0], inputVector[0]);
    for (int i = 0; i != node->featureDimension; ++i)
		node->referenceVector[i] += (inputVector[i] - node->referenceVector[i]) * changeRate;
}

NodeEdge NodeEdge_createEdge(
    NeuralGasNetwork* network,
	unsigned int n1,
	unsigned int n2,
	unsigned int age_); //Looks for the first non empty spot

void NodeEdge_deleteEdge();


kernel void calculateDistance(global NeuralGasNetwork* network, global float* inputVector)
{
    const int index = get_global_id(0);

    if (network->nodes[index].isNull)
    {
        network->nodes[index].distance = INT_MAX;
        network->nodes[index].error = 0; //for use in later stages
        return;
    }

    network->nodes[index].distance = 0; //Set the distance to zero
    for (unsigned int i = 0; i < 2; ++i)// network->nodes[index].featureDimension; ++i)
        network->nodes[index].distance += (network->nodes[index].referenceVector[i] - inputVector[i])
                                          * ((network->nodes[index].referenceVector[i] - inputVector[i])); 
    //printf("NODE INDEX: %d; FEATURE DIM, %d; DIST: %f; REFERNCE VECTOR[0]: %f\n", index, network->nodes[index].featureDimension, network->nodes[index].distance, inputVector[0]);
}

//Resets the array used to perform calculations
kernel void resetWorkIndexArray(global int* workIndexArray)
{
    const int index = get_global_id(0);
    workIndexArray[index] = index;
}

kernel void setWorkIndexArrayToZero(global int* workIndexArray)
{
    const int index = get_global_id(0);
    workIndexArray[index] = 0;
}

kernel void setWorkIndexArrayValue(global int* workIndexArray, int index, int value)
{
    workIndexArray[index] = value;
}

//No need for special code in case of an odd number of nodes to iterate over as in that case
//  it suffices to simply reduce the NDRange of the executed kernels by 1, effectively ignoring the
//  last node
kernel void findNearestNode(
    global NeuralGasNetwork* network,
    global int* workIndexArray,
    int spacing)
{
    const int index = 2 * spacing * get_global_id(0);
    global Node* node1 = &network->nodes[workIndexArray[index]];
    global Node* node2 = &network->nodes[workIndexArray[index + spacing]];
    if (node1->distance > node2->distance)
        workIndexArray[index] = workIndexArray[index + spacing];
}

//Given that all the distances have been calculted, find the two nodes with the smallest distance
//After log(n) calls of this kernel, the indexes will be located in workIndexArray[0,1]
kernel void findTwoNearestNode(
    global NeuralGasNetwork* network,
    global int* workIndexArray,
    int spacing)
{
    const int index = 2 * spacing * get_global_id(0);
    global Node* node3 = &network->nodes[workIndexArray[index + spacing]];
    global Node* node4 = &network->nodes[workIndexArray[index + spacing + 1]];

    if (network->nodes[workIndexArray[index]].distance >
         network->nodes[workIndexArray[index + 1]].distance)
    {
        int k = workIndexArray[index];
        workIndexArray[index] = workIndexArray[index + 1];
        workIndexArray[index + 1] = k;
    }
    //Guarantees workIndexArray[index] <= workIndexArray[index+1]
    if (node3->distance < network->nodes[workIndexArray[index]].distance)
    {
        workIndexArray[index + 1] = workIndexArray[index];
        workIndexArray[index] = workIndexArray[index + spacing];
    }
    else if (node3->distance <  network->nodes[workIndexArray[index + 1]].distance)
    {
        workIndexArray[index + 1] = workIndexArray[index + spacing];
    }
        
    if (node4->distance < network->nodes[workIndexArray[index]].distance)
    {
        workIndexArray[index + 1] = workIndexArray[index];
        workIndexArray[index] = workIndexArray[index + spacing + 1];
    }
    else if (node4->distance <  network->nodes[workIndexArray[index + 1]].distance)
    {
        workIndexArray[index + 1] = workIndexArray[index + spacing + 1];
    }
    //if (index == 0 && spacing == 512)
    //    printf("WORKINDEX[0].distance: %f, WORKINDEX[1].distance: %f \n", network->nodes[workIndexArray[0]].distance, network->nodes[workIndexArray[1]].distance );
}


//Work items spawned once
//**SINGLE THREADED**
kernel void iterate_step1(
    global NeuralGasNetwork* network,
    global int* workIndexArray,
    global float* inputVector)
{
   
    int nearestNodeIndex = workIndexArray[0];
    int secondNearestNodeIndex = workIndexArray[1];

    global Node* nearestNode = &network->nodes[nearestNodeIndex];
    global Node* secondNearestNode = &network->nodes[secondNearestNodeIndex];

    //Update error of nearest node
    nearestNode->error += nearestNode->distance;

    //Move the nearest node, and all the nodes connected to it
    shiftReferenceVector(nearestNode, inputVector, NODE_CHANGE_RATE);
}

//Increments the age of every edge by one except if it connects the two closest edge.
//  -In that case reset the age to zero
//  -Also if a node is neighboring, shift it
//If there are any edges with an age greater than AGE_MAX, delete it
kernel void iterate_step2(
    global NeuralGasNetwork* network,
    global int* workIndexArray,
    global float* inputVector)
{
    const int index = get_global_id(0);
    global NodeEdge* edge = &network->edges[index];
    if (!edge->isNull)
    {    
        if (edge->nodeIndex1 == workIndexArray[0] ) 
        {
            if (edge->nodeIndex2 == workIndexArray[1])
            {
                edge->age = 0;
                workIndexArray[MAX_NODES-1] = index;
            }
            shiftReferenceVector(&network->nodes[workIndexArray[1]], inputVector, NODE_NEIGHBOR_CHANGE_RATE);
        }
        else if (edge->nodeIndex2 == workIndexArray[0] ) 
        {
            if (edge->nodeIndex1 == workIndexArray[1])
            {
                edge->age = 0;
                workIndexArray[MAX_NODES-1] = index;
            }
            shiftReferenceVector(&network->nodes[workIndexArray[1]], inputVector, NODE_NEIGHBOR_CHANGE_RATE);
        }
        else
        {
            edge->age += 1;
            if (edge->age > AGE_MAX)
            {
                edge->isNull = true;
                //printf("HEYYYYYYYYYYYYYYYYYY: %d\n", index);
                IntStack_push(&network->emptyEdgeIndexStack, index);
            }
        }
    }
}

//Create an edge between the two closest nodes if no edge exists
//**SINGLE THREADED**
kernel void iterate_step3_addEdge(
    global NeuralGasNetwork* network,
    global int* workIndexArray)
{
    if (network->edges[workIndexArray[MAX_NODES-1]].age != 0) // no edge between two closest nodes
    {
        int nearestNodeIndex = workIndexArray[0];
        int secondNearestNodeIndex = workIndexArray[1];
        int nextIndex = IntStack_pop(&network->emptyEdgeIndexStack);
        network->edges[nextIndex].isNull = false;
        network->edges[nextIndex].nodeIndex1 = nearestNodeIndex;
        network->edges[nextIndex].nodeIndex2 = secondNearestNodeIndex;
        network->edges[nextIndex].age = 0;
    }
}


kernel void findTwoLargestError(
    global NeuralGasNetwork* network,
    global int* workIndexArray,
    int spacing)
{
    const int index = 2 * spacing * get_global_id(0);
    global Node* node3 = &network->nodes[workIndexArray[index + spacing]];
    global Node* node4 = &network->nodes[workIndexArray[index + spacing + 1]];

    if (network->nodes[workIndexArray[index]].error <
         network->nodes[workIndexArray[index + 1]].error)
    {
        int k = workIndexArray[index];
        workIndexArray[index] = workIndexArray[index + 1];
        workIndexArray[index + 1] = k;
    }
    //Guarantees workIndexArray[index] >= workIndexArray[index+1]
    if (node3->error > network->nodes[workIndexArray[index]].error)
    {
        workIndexArray[index + 1] = workIndexArray[index];
        workIndexArray[index] = workIndexArray[index + spacing];
    }
    else if (node3->error >  network->nodes[workIndexArray[index + 1]].error)
    {
        workIndexArray[index + 1] = workIndexArray[index + spacing];
    }
        
    if (node4->error > network->nodes[workIndexArray[index]].error)
    {
        workIndexArray[index + 1] = workIndexArray[index];
        workIndexArray[index] = workIndexArray[index + spacing + 1];
    }
    else if (node4->error >  network->nodes[workIndexArray[index + 1]].error)
    {
        workIndexArray[index + 1] = workIndexArray[index + spacing + 1];
    }
    //if (index == 0 && spacing == 512)
        //printf("WORKINDEX[0].error: %f, WORKINDEX[1].error: %f \n", network->nodes[workIndexArray[0]].error, network->nodes[workIndexArray[1]].error );
    
}
//If appropriate condition is met on the host CPU code, add a node
//Execute before deleting nodes as the order does not affect result,
//but it is important to maintain the state of the workIndexArray 
//for this step
//**SINGLE THREADED**
kernel void iterate_step3_addNode(
    global NeuralGasNetwork* network,
    global int* workIndexArray)
{
    //Delete the edge connecting the two closet node if one exists
    //if (network->edges[workIndexArray[MAX_NODES-1]].age == 0) // edge between two closest nodes
    //    network->edges[workIndexArray[MAX_NODES-1]].isNull = true;

    global Node* highestErrorNode = &network->nodes[workIndexArray[0]];
    global Node* secondHighestErrorNode = &network->nodes[workIndexArray[1]];

    //Create a new node that is placed in between the two closest nodes
    int newNodeIndex = IntStack_pop(&network->emptyNodeIndexStack);

    global Node* newNode = &network->nodes[newNodeIndex];
    newNode->isNull = false;
    newNode->featureDimension = highestErrorNode->featureDimension;
    //printf("NEW NODE INDEX: %d, highesterrornode, %d, %f \n", newNodeIndex, workIndexArray[0], highestErrorNode->error);

    for (unsigned int i = 0; i != newNode->featureDimension; ++i)
    {
        newNode->referenceVector[i] = 
            (highestErrorNode->referenceVector[i] + secondHighestErrorNode->referenceVector[i]) / 2;
    }

    //Adjust the errors appropriately
    newNode->error = highestErrorNode->error;
    highestErrorNode->error *= LOCAL_DECREASE_RATE;
    secondHighestErrorNode->error *= LOCAL_DECREASE_RATE;

    //Create the appropriate edges
    int newEdgeIndex1 = IntStack_pop(&network->emptyEdgeIndexStack);
    int newEdgeIndex2 = IntStack_pop(&network->emptyEdgeIndexStack);

    network->edges[newEdgeIndex1].isNull = false;
    network->edges[newEdgeIndex1].age = 0;
    network->edges[newEdgeIndex1].nodeIndex1 = newNodeIndex;
    network->edges[newEdgeIndex1].nodeIndex2 = workIndexArray[0];

    network->edges[newEdgeIndex2].isNull = false;
    network->edges[newEdgeIndex2].age = 0;
    network->edges[newEdgeIndex2].nodeIndex1 = newNodeIndex;
    network->edges[newEdgeIndex2].nodeIndex2 = workIndexArray[1];
}

//No longer need the workIndexArray
//Finds the nodes that have no edges
kernel void iterate_step4(
    global NeuralGasNetwork* network,
    global int* workIndexArray)
{
    const int index = get_global_id(0);
    if (!network->edges[index].isNull)
    {
        workIndexArray[network->edges[index].nodeIndex1] = OPENCL_TRUE;
        workIndexArray[network->edges[index].nodeIndex2] = OPENCL_TRUE;
    }
}

//Deletes the nodes that have no edges
kernel void iterate_step5(
    global NeuralGasNetwork* network,
    global int* workIndexArray)
{
    const int index = get_global_id(0);
    bool nodeHasNoEdges = !(workIndexArray[index] == OPENCL_TRUE) &&
                            !network->nodes[index].isNull;
    if (nodeHasNoEdges)
    {
        network->nodes[index].isNull = true;
        network->nodes[index].distance = -1;
        //printf("HEYYYYYYYYYYYYYYYYYY: %d\n", index);
        IntStack_push(&network->emptyNodeIndexStack, index);
    }
}


//Globally decrease all the errors
kernel void iterate_step6(global NeuralGasNetwork* network)
{
    const int index = get_global_id(0);
    network->nodes[index].error *= GLOBAL_DECREASE_RATE;
}
