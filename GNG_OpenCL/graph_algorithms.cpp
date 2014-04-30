#include "graph_algorithms.h"

namespace CPU_serial_gng
{
    void LabelGraphNodes(NeuralGasNetwork* network)
    {
        //Reset all the classes to class -1
        for (auto nodeIt = network->nodes.begin(); nodeIt != network->nodes.end(); ++nodeIt)
            (*nodeIt)->classification = -1;

        //Perform a recursive DFS on the nodes
        int currentClass = 0;
        for (unsigned int i = 0; i != network->nodes.size(); ++i)
        {
            NeuralGasNode* curNode = network->nodes[i];
            if (curNode->classification == -1) // if we have not visited this node
            {
                DFS_Visit(curNode, currentClass); // visit it
                network->numberOfClassifications = currentClass + 1;
                ++currentClass; // we have visited all nodes connected to curNode; increment class counter
            }
        }
        cout << "Number of clusters detected: " << currentClass << endl;
    }

    void DFS_Visit(NeuralGasNode* node, int currentClass)
    {
        //Recursive DFS search
        node->classification = currentClass;
		cout << "class: " << currentClass  << " #edges: " << node->edges.size() << endl;
        for (unsigned int i = 0; i != node->edges.size(); ++i)
        {
            NeuralGasNode* neighbor = node->edges[i]->neighboringNode(node);
            if (neighbor->classification == -1) // if we have not visited this node
                DFS_Visit(neighbor, currentClass); // visit it
        }
    }
}