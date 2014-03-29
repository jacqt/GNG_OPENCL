#include "opencl_gng_network.h"

kernel void calculateDistance(global NeuralGasNetwork* network, global float* inputVector)
{
    const int index = get_global_id(0);
    global Node* curNode = & (network->nodes[index]);

    curNode->distance = 0; //Set the distance to zero
    for (unsigned int i = 0; i != curNode->featureDimension; ++i)
        curNode->distance += (curNode->referenceVector[i]) * (curNode->referenceVector[i]);
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

//Given that all the distances have been calculted, find the one with the smallest distance
//After log(n) calls of this kernel, the index will be located in workIndexArray[0]
//No need for special code in case of an odd number of nodes to iterate over as in that case
//  it suffices to simply reduce the NDRange of the executed kernels by 1, effectively ignoring the
//  last node
kernel void findNearestNodeKernel(
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
kernel void findTwoNearestNodeKernel(
    global NeuralGasNetwork* network,
    global int* workIndexArray,
    int spacing)
{
    const int index = 4 * spacing * get_global_id(0);
    global Node* node3 = &network->nodes[workIndexArray[index + 2 * spacing]];
    global Node* node4 = &network->nodes[workIndexArray[index + 2 * spacing + 1]];

    if (network->nodes[workIndexArray[index]].distance >
         network->nodes[workIndexArray[index + 1]].distance)
    {
        int k = workIndexArray[index];
        workIndexArray[index] = workIndexArray[index + 1];
        workIndexArray[index + 1] = k;
    }

    if (node3->distance < min(workIndexArray[index], workIndexArray[index + 1]))
    {
        workIndexArray[index + 1] = workIndexArray[index];
        workIndexArray[index] = workIndexArray[index + 2 * spacing];
    }
    else if (node3->distance < max(workIndexArray[index], workIndexArray[index + 1]))
    {
        workIndexArray[index + 1] = workIndexArray[index + 2 * spacing];
    }
        
    if (node4->distance < min(workIndexArray[index], workIndexArray[index + 1]))
    {
        workIndexArray[index + 1] = workIndexArray[index];
        workIndexArray[index] = workIndexArray[index + 2 * spacing + 1];
    }
    else if (node4->distance < max(workIndexArray[index], workIndexArray[index + 1]))
    {
        workIndexArray[index + 1] = workIndexArray[index + 2 * spacing + 1];
    }
}


//Work items spawned for each edge
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
//If there are any edges with an age greater than AGE_MAX, delete it
kernel void iterate_step2(
    global NeuralGasNetwork* network,
    global int* workIndexArray)
{
    const int index = get_global_id(0);
    global NodeEdge* edge = &network->edges[index];
    if (!edge->isNull)
    {    
        if ((edge->nodeIndex1 == workIndexArray[0] && edge->nodeIndex2 == workIndexArray[1]) ||
            (edge->nodeIndex1 == workIndexArray[1] && edge->nodeIndex2 == workIndexArray[0]))
        {
            edge->age = 0;
            workIndexArray[MAX_NODES-1] = index;
        }
        else
        {
            edge->age += 1;
            if (edge->age > AGE_MAX)
            {
                edge->isNull = true;
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
    if (workIndexArray[MAX_NODES-1] == 0) // no edge between two closest nodes
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
    if (workIndexArray[MAX_NODES-1] != 0) // edge between two closest nodes
        network->edges[workIndexArray[MAX_NODES-1]].isNull = true;

    global Node* nearestNode = &network->nodes[workIndexArray[0]];
    global Node* secondNearestNode = &network->nodes[workIndexArray[1]];

    //Create a new node that is placed in between the two closest nodes
    int newNodeIndex = IntStack_pop(&network->emptyNodeIndexStack);
    global Node* newNode = &network->nodes[newNodeIndex];
    newNode->isNull = false;

    for (unsigned int i = 0; i != newNode->featureDimension; ++i)
    {
        newNode->referenceVector[i] = 
            (nearestNode->referenceVector[i] + secondNearestNode->referenceVector[i]) / 2;
    }

    //Adjust the errors appropriately
    newNode->error = nearestNode->error;
    nearestNode->error *= LOCAL_DECREASE_RATE;
    secondNearestNode->error *= LOCAL_DECREASE_RATE;

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

//Finds the nodes that have no edges
kernel void iterate_step4(
    global NeuralGasNetwork* network,
    global int* workIndexArray)
{
    const int index = get_global_id(0);
    workIndexArray[network->edges[index].nodeIndex1] = CL_TRUE;
    workIndexArray[network->edges[index].nodeIndex2] = CL_TRUE;
}

//Deletes the nodes that have no edges
kernel void iterate_step5(
    global NeuralGasNetwork* network,
    global int* workIndexArray)
{
    const int index = get_global_id(0);
    bool nodeHasNoEdges = !(workIndexArray[index] == CL_TRUE);
    if (nodeHasNoEdges)
    {
        network->nodes[index].isNull = true;
        network->nodes[index].distance = -1;
        IntStack_push(&network->emptyNodeIndexStack, index);
    }
}


//Globally decrease all the errors
kernel void iterate_step6(global NeuralGasNetwork* network)
{
    const int index = get_global_id(0);
    network->nodes[index].error *= GLOBAL_DECREASE_RATE;
}
