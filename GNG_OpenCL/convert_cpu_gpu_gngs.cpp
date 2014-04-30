#include "convert_cpu_gpu_gngs.h"

/*
GPU_parallel_gng::NeuralGasNetwork* convert_to_gpu_gng(CPU_serial_gng::NeuralGasNetwork &originalNet)
{
    //
}
*/


//Posible algorithms: create a mapping between the origianl index in the array to the new position
//in the CPU_serial_gng::NeuralGasNetwork->nodes
CPU_serial_gng::NeuralGasNetwork* convert_to_cpu_gng(GPU_parallel_gng::NeuralGasNetwork &originalNet)
{
	CPU_serial_gng::NeuralGasNetwork* newNet = new CPU_serial_gng::NeuralGasNetwork(
        originalNet.nodes[0].featureDimension,
        NODE_CHANGE_RATE,
        NODE_NEIGHBOR_CHANGE_RATE,
        LOCAL_DECREASE_RATE,
        GLOBAL_DECREASE_RATE,
        AGE_MAX,
        TIME_BETWEEN_ADDING_NODES);

    //Now delete all the edges and nodes.
	newNet->nodes.erase(newNet->nodes.begin(), newNet->nodes.end());
	newNet->edges.erase(newNet->edges.begin(), newNet->edges.end());

    vector<int> mapping; //maps the ith node in the originalNet.nodes to newNet.nodes
    int j = 0;
    for (unsigned  int i = 0; i != MAX_NODES; ++i)
	{
		if (originalNet.nodes[i].isNull)
		{
            mapping.push_back(-1);
            continue;
		}

        vector<double> newReferenceVector;
        for (unsigned int j = 0; j != originalNet.nodes[i].featureDimension; ++j)
            newReferenceVector.push_back(originalNet.nodes[i].referenceVector[j]);

        CPU_serial_gng::NeuralGasNode* newNode = new CPU_serial_gng::NeuralGasNode(newReferenceVector);
        newNode->error = originalNet.nodes[i].error;
        newNode->nodeIndex = j;
        newNet->nodes.push_back(newNode);
        mapping.push_back(j);
        ++j;
	}

    for (unsigned int i = 0; i != MAX_EDGES; ++i)
	{
        if (originalNet.edges[i].isNull)
            continue;
        cout << "AN EDGE: " << originalNet.edges[i].nodeIndex1 << " " << originalNet.edges[i].nodeIndex2 << endl;
        CPU_serial_gng::NeuralGasNode* n1;
        CPU_serial_gng::NeuralGasNode* n2;
        n1 = newNet->nodes[mapping[originalNet.edges[i].nodeIndex1]];
        n2 = newNet->nodes[mapping[originalNet.edges[i].nodeIndex2]];
        
        CPU_serial_gng::NodeEdge* newEdge = new CPU_serial_gng::NodeEdge(n1,n2,originalNet.edges[i].age);
        newEdge->age = originalNet.edges[i].age; //possibly unnecessary
        newNet->edges.push_back(newEdge);
        n1->edges.push_back(newEdge);
        n2->edges.push_back(newEdge);
	}
	cout << "# Nodes: " << newNet->nodes.size() << "  # Edges: " << newNet->edges.size() << endl;

    return newNet;

}