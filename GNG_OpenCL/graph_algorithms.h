#ifndef GRAPH_ALGORITHMS_H
#define GRAPH_ALGORITHMS_H

#include "general_include.h"
#include "cpu_gng.h"


namespace CPU_serial_gng
{
    void LabelGraphNodes(NeuralGasNetwork* network);

    void DFS_Visit(NeuralGasNode* node, int currentClass);
}


#endif