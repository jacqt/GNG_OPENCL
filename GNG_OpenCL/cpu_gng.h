#ifndef CPU_GNG_H
#define CPU_GNG_H

#include "general_include.h"
#include "utils.h"

namespace CPU_serial_gng
{
    class NodeEdge;
    class NeuralGasNetwork;
    class NeuralGasNode;

    class NodeEdge{
    public:
        NeuralGasNode* node1;
        NeuralGasNode* node2;
        double age;

        NodeEdge(NeuralGasNode* n1, NeuralGasNode* n2, double age_);

        NeuralGasNode* neighboringNode(NeuralGasNode* node);

        ~NodeEdge();
    };


    class NeuralGasNode {
    public:
        double error; //Local error of the node
        unsigned int nodeIndex; //Index in the NeuralGasNetwork node array - not automatically updated
        unsigned int classification; // Integer to represent the classification of the node
        vector<double> referenceVector; //vector describing the node's location in the feauture space
        vector<NodeEdge*> edges;

        NeuralGasNode(const vector<double> &initialRefVector);

        //Set error += weightChange
        void inline updateError(double weightChange);

        //Returns the squared euclidan distance between the reference vector and the feature vector
        double getEuclidianDistance(const vector<double> &featurevector);

        //adjust the node and it's neighbor nodes reference vectors
        void shiftReferenceVector(const vector<double> &dest, double changeAmount);

    private:
        //
        //
    };

    class NeuralGasNetwork{
    public:
        unsigned int numberOfClassifications;
        vector<NeuralGasNode*> nodes;
        vector<NodeEdge*> edges;

        //Given a featuer space dimension, initializes two random points
        //Assume that each vector feature is spaced around [0.0, 1.0]
        NeuralGasNetwork(
            unsigned int featureSpaceDimension,
            double newNodeChangeRate,
            double newNodeNeighborChangeRate,
            double newErrorDecreaseFactorLocal,
            double newErrorDecreaseFactorGlobal,
            unsigned int newAgeMax,
            unsigned int newTimeBetweenAddingNodes);
        
        //Given a filename, load a GNG
        NeuralGasNetwork(
            const string fileName,
            unsigned int featureSpaceDimension,
            double newNodeChangeRate,
            double newNodeNeighborChangeRate,
            double newErrorDecreaseFactorLocal,
            double newErrorDecreaseFactorGlobal,
            unsigned int newAgeMax,
            unsigned int newTimeBetweenAddingNodes);

        ~NeuralGasNetwork();

        //Add a node to the network
        void AddNode();

        //Finds two closest nodes to a feature vector in terms of the euclidian distance
        void findTwoNearest(const vector<double> &featureVector,
            unsigned int &closest,
            unsigned int &nextClosest);

        //Finds the cloest node to a feature vector in terms of euclidian distances
        unsigned int findNearest(const vector<double> &featureVector);

        //returns the index of the node with the largest error
        unsigned int findNodeWithLargestError();

        //Iterate over a data sample
        void iterate(const vector<double> &featureVector);

        //Write the network to a file
        void writeToFile();

        //Load the network from a file
        void readFromFile(const string fileNme);

        //Set the network's maximum number of nodes
        void setMaxNodes(unsigned int newMaxNodes);


    private:
        double nodeChangeRate;
        double nodeNeighborChangeRate;
        double errorDecreaseFactorLocal;
        double errorDecreaseFactorGlobal;
        unsigned int ageMax;
        unsigned int timeBetweenAddingNodes;
        unsigned int currentIteration;
        unsigned int maxNodes;

        void deleteEdge(unsigned int edgeIndex);

        void deleteEdge(NodeEdge* edge);

        void deleteNode(unsigned int nodeIndex);
    };

}
#endif