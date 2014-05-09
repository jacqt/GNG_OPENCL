#ifndef GPU_GNG_NETWORK_H
#define GPU_GNG_NETWORK_H

#include "general_include.h"
#include "utils.h"

namespace GPU_parallel_gng
{
    class NeuralGasNetworkHost;
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

    //Constructors for the node object
    Node Node_createNewNode(unsigned int _featureDimension);
    Node Node_createNewNode(vector<float> _referenceVector, unsigned int _error);

    //Constructors for the edge object
    NodeEdge NodeEdge_createNewNodeEdge(
        unsigned int _nodeIndex1,
        unsigned int _nodeIndex2,
        unsigned int _age = 0);

    void NeuralGasNetwork_addNode(NeuralGasNetwork* network, Node* node);
    void NeuralGasNetworkHost_addNode(
        NeuralGasNetwork* network,
        vector<float> _referenceVector,
        unsigned int _error);
    void NeuralGasNetwork_addNodeEdge(NeuralGasNetwork* network, NodeEdge* edge);
    void NeuralGasNetworkHost_addNodeEdge(
        NeuralGasNetwork* network,
        unsigned int _nodeIndex1,
        unsigned int _nodeIndex2,
        unsigned int _age);

    NeuralGasNetwork* NeuralGasNetwork_createNewGNG(unsigned int featureDimension);

    class NeuralGasNetworkHost {
    public:
        NeuralGasNetwork* gngNetwork; // the network struct

        //Given a featuer space dimension, initializes two random points
        //Assume that each vector feature is spaced around [0.0, 1.0]
        NeuralGasNetworkHost(
            unsigned int featureSpaceDimension,
            double newNodeChangeRate,
            double newNodeNeighborChangeRate,
            double newErrorDecreaseFactorLocal,
            double newErrorDecreaseFactorGlobal,
            unsigned int newAgeMax,
            unsigned int newTimeBetweenAddingNodes,
            cl::Context &context,
            cl::Program &program,
            cl::CommandQueue &queue);

        //Given a filename, load a GNG
        NeuralGasNetworkHost(
            const string fileName,
            unsigned int featureSpaceDimension,
            double newNodeChangeRate,
            double newNodeNeighborChangeRate,
            double newErrorDecreaseFactorLocal,
            double newErrorDecreaseFactorGlobal,
            unsigned int newAgeMax,
            unsigned int newTimeBetweenAddingNodes);

        //Create the memory buffers
        void createMemoryBuffers(cl::Context &context);

        //Create the kernels
        void createKernels(cl::Program &program);

        //Writes the network to a file
        void writeNetworkToFile(string fileName);

        //Iterate the network once given a training point
        void iterateNetwork(
            vector<float> &trainingPoint,
            cl::CommandQueue &queue);

        //Traing the network given the data and number of iterations
        void trainNetwork(
            vector<vector<float>> &trainingData,
            cl::CommandQueue &queue,
            unsigned int trainingIterations);

        //Trains the network given a filename, a parser, and number of iterations
        void trainNetwork(
            string &fileName,
            void (*parser) (string &inputString, vector<float> &outputVector),
            cl::CommandQueue &queue,
            unsigned int trainingIterations);

        //Does random things with the network
        void foo(cl::CommandQueue &queue);

    private:
        vector<float> inputArray;
        vector<int> workIndexArray;
        cl::Buffer gngNetworkBuffer; // buffer of the network struct on the GPU
        cl::Buffer inputBuffer; // buffer that holds the inputs on the GPU
        cl::Buffer workIndexBuffer; // used in parallelization certain sequential algorithms
        unsigned int sizeOfgngNetworkBuffer;
        unsigned int sizeOfInputBuffer;
        unsigned int sizeOfWorkIndexBuffer;
        cl::Kernel calculateDistanceKernel;
        cl::Kernel resetWorkIndexArrayKernel;
        cl::Kernel setWorkIndexArrayToZeroKernel;
        cl::Kernel setWorkIndexArrayValueKernel;
        cl::Kernel findNearestNodeKernel;
        cl::Kernel findTwoNearestNodeKernel;
        cl::Kernel findTwoLargestErrorKernel; //Variable NDRange
        cl::Kernel findNeighborKernel; //NDRange: MAX_EDGES
        cl::Kernel iterate_step1_Kernel; //NDRange : 1
        cl::Kernel iterate_step2_Kernel; //NDRange : MAX_EDGES
        cl::Kernel iterate_step3_addEdge_Kernel; //NDRange : 1
        cl::Kernel iterate_step3_addNode_Kernel; //NDRange : 1
        cl::Kernel iterate_step4_Kernel; //NDRange : MAX_EDGES
        cl::Kernel iterate_step5_Kernel; //NDRange : MAX_NODES
        cl::Kernel iterate_step6_Kernel; //NDRange : MAX_NODES

        unsigned int iterationNumber;
    };
}
#endif