#include "gpu_gng.h"

namespace GPU_parallel_gng
{
    /////////////////////////////////////////////////////////////////////////
    //                Functions related to the struct                      //
    /////////////////////////////////////////////////////////////////////////

    //Creates a node
    Node Node_createNewNode(unsigned int _featureDimension)
    {
        Node newNode;
        newNode.isNull = false;
        newNode.error = 0;
        newNode.featureDimension = _featureDimension;
        newNode.distance = 0;
        for (unsigned int i = 0; i != _featureDimension; ++i)
            newNode.referenceVector[i] = getRandomFloat(0,1);
        return newNode;
    }

    //Creates a NodeEdge
    NodeEdge NodeEdge_createNewNodeEdge(
        unsigned int _nodeIndex1,
        unsigned int _nodeIndex2,
        unsigned int _age)
    {
        NodeEdge newEdge;
        newEdge.isNull = false;
        newEdge.nodeIndex1 = _nodeIndex1;
        newEdge.nodeIndex2 = _nodeIndex2;
        newEdge.age = _age;
        return newEdge;
    }

    //Creates the network, and adds the initial two random nodes
    NeuralGasNetwork* NeuralGasNetwork_createNewGNG(unsigned int featureDimension)
    {
        NeuralGasNetwork* network = new NeuralGasNetwork;
        network->emptyNodeIndexStack = createNewStack(MAX_NODES);
        network->emptyEdgeIndexStack = createNewStack(MAX_EDGES);

        for (unsigned int i = 0; i != MAX_NODES; ++i)
        {
            network->nodes[i].isNull = true;
            //set distance to negative one in order to get around problems in finding the closest node
            //to a particular feature vector
            network->nodes[i].distance = -1; 
        }
        for (unsigned int i = 0; i != MAX_EDGES; ++i)
            network->edges[i].isNull = true;

        //Two initial nodes
        network->nodes[0] = Node_createNewNode(featureDimension);
        network->nodes[1] = Node_createNewNode(featureDimension);

        //Create an edge between them
        network->edges[0] = NodeEdge_createNewNodeEdge(0,1);

        return network;
    }

    void NeuralGasNetwork_addNode(NeuralGasNetwork* network, Node* node)
    {
        int nextIndex = IntStack_pop(&network->emptyNodeIndexStack);
        network->nodes[nextIndex] = *node;
    }

    void NeuralGasNetwork_addNodeEdge(NeuralGasNetwork* network, NodeEdge* edge)
    {
        int nextIndex = IntStack_pop(&network->emptyEdgeIndexStack);
        network->edges[nextIndex] = *edge;
    }

    void NeuralGasNetworkHost_addNode(
        NeuralGasNetwork* network,
        vector<float> _referenceVector,
        unsigned int _error)
    {
        int nextIndex = IntStack_pop(&network->emptyNodeIndexStack);
        network->nodes[nextIndex].isNull = false;
        network->nodes[nextIndex].error = _error;
        for (unsigned int i = 0; i != _referenceVector.size(); ++i)
            network->nodes[nextIndex].referenceVector[i] = _referenceVector[i];
    }
    void NeuralGasNetworkHost_addNodeEdge(
        NeuralGasNetwork* network,
        unsigned int _nodeIndex1,
        unsigned int _nodeIndex2,
        unsigned int _age)
    {
        int nextIndex = IntStack_pop(&network->emptyEdgeIndexStack);
        network->edges[nextIndex].isNull = false;
        network->edges[nextIndex].nodeIndex1 = _nodeIndex1;
        network->edges[nextIndex].nodeIndex2 = _nodeIndex2;
    }



    /////////////////////////////////////////////////////////////////////////
    //                       GNG Host CPU code                             //
    /////////////////////////////////////////////////////////////////////////

    NeuralGasNetworkHost::NeuralGasNetworkHost(
        unsigned int featureDimension,
        double newNodeChangeRate,
        double newNodeNeighborChangeRate,
        double newErrorDecreaseFactorLocal,
        double newErrorDecreaseFactorGlobal,
        unsigned int newAgeMax,
        unsigned int newTimeBetweenAddingNodes,
        cl::Context &context,
        cl::Program &program,
        cl::CommandQueue &queue)
    {
        //Create the Neural Gas Network struct
        gngNetwork = NeuralGasNetwork_createNewGNG(featureDimension);
        for (unsigned int i = 0; i != featureDimension; ++i)
            inputArray.push_back(0);
        for (unsigned int i = 0; i != MAX_NODES; ++i)
            workIndexArray.push_back(0);
        inputArray.shrink_to_fit();
        workIndexArray.shrink_to_fit();
        createMemoryBuffers(context);
        createKernels(program);
        //Transfer the GNG to the GPU??
        queue.enqueueWriteBuffer(gngNetworkBuffer,CL_TRUE, 0, sizeOfgngNetworkBuffer, gngNetwork);
          //  sizeOfgngNetworkBuffer, gngNetwork);

    }

    //Given a filename, load a GNG
    NeuralGasNetworkHost::NeuralGasNetworkHost(
        const string fileName,
        unsigned int featureSpaceDimension,
        double newNodeChangeRate,
        double newNodeNeighborChangeRate,
        double newErrorDecreaseFactorLocal,
        double newErrorDecreaseFactorGlobal,
        unsigned int newAgeMax,
        unsigned int newTimeBetweenAddingNodes)
    {

     
    }


    void NeuralGasNetworkHost::createMemoryBuffers(cl::Context &context)
    {
        sizeOfgngNetworkBuffer = sizeof(*gngNetwork);
        sizeOfInputBuffer = sizeof(inputArray);
        sizeOfWorkIndexBuffer = sizeof(workIndexArray);

        gngNetworkBuffer = cl::Buffer(context, CL_MEM_READ_WRITE  | CL_MEM_COPY_HOST_PTR,
            sizeOfgngNetworkBuffer, gngNetwork);

        inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
            sizeOfInputBuffer, &inputArray[0]);

        workIndexBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            sizeOfWorkIndexBuffer, &workIndexArray[0]);
    }

    void NeuralGasNetworkHost::createKernels(cl::Program &program)
    {
        calculateDistanceKernel = cl::Kernel(program, "calculateDistance");
        resetWorkIndexArrayKernel = cl::Kernel(program, "resetWorkIndexArray");
        setWorkIndexArrayToZeroKernel = cl::Kernel(program, "setWorkIndexArrayToZero");
        setWorkIndexArrayValueKernel = cl::Kernel(program, "setWorkIndexArrayValue");
        findNearestNodeKernel = cl::Kernel(program, "findNearestNode");
        findTwoNearestNodeKernel = cl::Kernel(program,"findTwoNearestNode");
        findTwoLargestErrorKernel = cl::Kernel(program, "findTwoLargestError");
        iterate_step1_Kernel = cl::Kernel(program, "iterate_step1");
        iterate_step2_Kernel = cl::Kernel(program, "iterate_step2");
        iterate_step3_addEdge_Kernel = cl::Kernel(program, "iterate_step3_addEdge");
        iterate_step3_addNode_Kernel = cl::Kernel(program, "iterate_step3_addNode");
        iterate_step4_Kernel = cl::Kernel(program, "iterate_step4");
        iterate_step5_Kernel = cl::Kernel(program, "iterate_step5");
        iterate_step6_Kernel = cl::Kernel(program, "iterate_step6");

        calculateDistanceKernel.setArg(0, gngNetworkBuffer);
        calculateDistanceKernel.setArg(1, inputBuffer);

        resetWorkIndexArrayKernel.setArg(0, workIndexBuffer);

        setWorkIndexArrayToZeroKernel.setArg(0, workIndexBuffer);

        setWorkIndexArrayValueKernel.setArg(0, workIndexBuffer);
      //setWorkIndexArrayValueKernel.setArg(1, int index);
      //setWorkIndexArrayValueKernel.setArgwhat (2, int value);

        findNearestNodeKernel.setArg(0, gngNetworkBuffer);
        findNearestNodeKernel.setArg(1, workIndexBuffer);
      //findNearestNodeKernel.setArg(2, int spacing);

        findTwoNearestNodeKernel.setArg(0, gngNetworkBuffer);
        findTwoNearestNodeKernel.setArg(1, workIndexBuffer);
      //findTwoNearestNodeKernel.setArg(2, int spacing);

        findTwoLargestErrorKernel.setArg(0, gngNetworkBuffer);
        findTwoLargestErrorKernel.setArg(1, workIndexBuffer);
      //findTwoLargestErrorKernel.setArg(2, int spacing);

        iterate_step1_Kernel.setArg(0, gngNetworkBuffer);
        iterate_step1_Kernel.setArg(1, workIndexBuffer);
        iterate_step1_Kernel.setArg(2, inputBuffer);

        iterate_step2_Kernel.setArg(0, gngNetworkBuffer);
        iterate_step2_Kernel.setArg(1, workIndexBuffer);
        iterate_step2_Kernel.setArg(2, inputBuffer);

        iterate_step3_addEdge_Kernel.setArg(0, gngNetworkBuffer);
        iterate_step3_addEdge_Kernel.setArg(1, workIndexBuffer);

        iterate_step3_addNode_Kernel.setArg(0, gngNetworkBuffer);
        iterate_step3_addNode_Kernel.setArg(1, workIndexBuffer);

        iterate_step4_Kernel.setArg(0, gngNetworkBuffer);
        iterate_step4_Kernel.setArg(1, workIndexBuffer);

        iterate_step5_Kernel.setArg(0, gngNetworkBuffer);
        iterate_step5_Kernel.setArg(1, workIndexBuffer);

        iterate_step6_Kernel.setArg(0, gngNetworkBuffer);
    }

    void NeuralGasNetworkHost::writeNetworkToFile(string fileName)
    {

    }

    void NeuralGasNetworkHost::iterateNetwork(
        vector<float> &trainingPoint,
        cl::CommandQueue &queue)
    {
        //Reset the work index array
        queue.enqueueNDRangeKernel(
            resetWorkIndexArrayKernel, 
            cl::NullRange,
            cl::NDRange(MAX_NODES),
            cl::NullRange);

        //Calculate the squared euclidian distances
        queue.enqueueWriteBuffer(inputBuffer,CL_TRUE, 0, sizeOfInputBuffer, &trainingPoint[0]);
        //cout << trainingPoint[0] << " " << trainingPoint[1] << endl;
        queue.enqueueNDRangeKernel(
            calculateDistanceKernel, 
            cl::NullRange, 
            cl::NDRange(MAX_NODES), 
            cl::NullRange);

        //Find the two nearest nodes
        int spacing = 2;
        int range = MAX_NODES/4;
        //cout << spacing << " " << range << endl;
        while (spacing < MAX_NODES)
        {
            //cout << spacing << " " << range << endl;
            findTwoNearestNodeKernel.setArg(2, spacing);
                queue.enqueueNDRangeKernel(
                    findTwoNearestNodeKernel,
                    cl::NullRange,
                    cl::NDRange(range),
                    cl::NullRange);
            spacing *= 2;
            range = range/2;
        }

        //Iterate steps..
        queue.enqueueNDRangeKernel(
            iterate_step1_Kernel,
            cl::NullRange,
            cl::NDRange(1),
            cl::NullRange);

        queue.enqueueNDRangeKernel(
            iterate_step2_Kernel,
            cl::NullRange,
            cl::NDRange(MAX_EDGES),
            cl::NullRange);

        queue.enqueueNDRangeKernel(
            iterate_step3_addEdge_Kernel,
            cl::NullRange,
            cl::NDRange(1),
            cl::NullRange);

        queue.enqueueNDRangeKernel(
            iterate_step4_Kernel,
            cl::NullRange,
            cl::NDRange(MAX_EDGES),
            cl::NullRange);

        queue.enqueueNDRangeKernel(
            iterate_step5_Kernel,
            cl::NullRange,
            cl::NDRange(MAX_NODES),
            cl::NullRange);

        queue.enqueueNDRangeKernel(
            iterate_step6_Kernel,
            cl::NullRange,
            cl::NDRange(MAX_NODES),
            cl::NullRange);

        if (iterationNumber % TIME_BETWEEN_ADDING_NODES == 0) // Add a node
        {
            //No longer need results from workIndexArray:
            queue.enqueueNDRangeKernel(
                resetWorkIndexArrayKernel, 
                cl::NullRange,
                cl::NDRange(MAX_NODES),
                cl::NullRange);

            //Find the two nodes with the highest error
            int spacing = 2;
            int range = MAX_NODES/4;
            while (spacing < MAX_NODES)
            {
                findTwoLargestErrorKernel.setArg(2, spacing);
                queue.enqueueNDRangeKernel(
                    findTwoLargestErrorKernel,
                    cl::NullRange,
                    cl::NDRange(range),
                    cl::NullRange);
                spacing *= 2;
                range = range/2;
            }

            cout << "ADDING" << endl;
            queue.enqueueNDRangeKernel(
                iterate_step3_addNode_Kernel,
                cl::NullRange,
                cl::NDRange(1),
                cl::NullRange);
        }
        ++iterationNumber;
    }


    void NeuralGasNetworkHost::foo(cl::CommandQueue &queue)
    {
        queue.enqueueReadBuffer(gngNetworkBuffer, CL_TRUE, 0, sizeOfgngNetworkBuffer,gngNetwork);
        for (int i = 0; i != MAX_NODES; ++i)
        {
            Node* node = &gngNetwork->nodes[i];
            if (node->isNull)
                continue;
            cout << node->referenceVector[0] << ", " << node->referenceVector[1] << endl;
        }
    }

    void NeuralGasNetworkHost::trainNetwork(
        vector<vector<float>> &trainingData,
        cl::CommandQueue &queue,
        unsigned int trainingIterations)
    {

    }

    void NeuralGasNetworkHost::trainNetwork(
        string &fileName,
        void (*parser) (string &inputString, vector<float> &outputVector),
        cl::CommandQueue &queue,
        unsigned int trainingIterations)
    {
        iterationNumber = 1;

    }
}