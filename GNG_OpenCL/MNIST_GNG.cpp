#include "MNIST_GNG.h"
#define MNIST_GNG_FILE          "GNG-NN.net"
//#define DATA_RANGE          60000

int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

void readMNIST(vector<double*> &inputs, vector<int> &targets, int &inputSize)
{
    //Read the targets
    std::ifstream targetFile("data/train-labels.idx1-ubyte", std::ios::binary);
    if (targetFile.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        targetFile.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        targetFile.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
#ifndef DATA_RANGE
        for (int i = 0; i != number_of_images; ++i)
#else
        for (int i = 0; i != DATA_RANGE; ++i)
#endif
        {
            unsigned char temp = 0;
            targetFile.read((char*) &temp, sizeof(temp));
            targets.push_back((int) temp);

            if (i % 10000 == 0)
                cout << i << "/" << number_of_images << endl;
        }
    }
    std::ifstream trainingFeatureFile("data/train-images.idx3-ubyte", std::ios::binary);
    if (trainingFeatureFile.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        trainingFeatureFile.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        trainingFeatureFile.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);

        trainingFeatureFile.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);

        trainingFeatureFile.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        inputSize = n_rows * n_cols;
#ifndef DATA_RANGE
        for (int i = 0; i != number_of_images; ++i)
#else
        for (int i = 0; i != DATA_RANGE; ++i)
#endif
        {
            double* inputData = new double[n_rows*n_cols];
            for (int r = 0; r<n_rows; ++r)
            {
                for (int c = 0; c<n_cols; ++c)
                {
                    unsigned char temp = 0;
                    trainingFeatureFile.read((char*) &temp, sizeof(temp));
                    inputData[r*n_cols + c] = ((float) temp)/25.0;
                }
            }
            inputs.push_back(inputData);
            if (i % 10000 == 0)
                cout << i << "/" << number_of_images << endl;
        }
    }
    inputs.shrink_to_fit();
    targets.shrink_to_fit();
}

void trainMNIST_GNG()
{
    vector<double*> inputs;
    vector<int> targets;
    int inputSize;
    cout << "Reading MNIST data set" << endl;
    readMNIST(inputs, targets, inputSize);
    cout << "Finished reading. Now creating neural net" << endl;
    cout << endl;

        //Get the platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    //Create a context
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*platforms.begin())(), 0};
    cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

    //Create and build the program
    cl::Program opencl_gngNetworkProgram;
    opencl_gngNetworkProgram = createProgram(context, "opencl_gng_network.cl");

    //Create the command queue from the first device in context
    cl::CommandQueue queue;
    queue = cl::CommandQueue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], CL_QUEUE_PROFILING_ENABLE);


    GPU_parallel_gng::NeuralGasNetworkHost* myNet = new GPU_parallel_gng::NeuralGasNetworkHost(
        784,
        NODE_CHANGE_RATE,
        NODE_NEIGHBOR_CHANGE_RATE,
        LOCAL_DECREASE_RATE,
        GLOBAL_DECREASE_RATE,
        AGE_MAX,
        TIME_BETWEEN_ADDING_NODES,
        context,
        opencl_gngNetworkProgram,
        queue);


    int i = 0;
    while (i < 50000000) //50 MIL.
    {
        if (i % 10000 == 0)
        {
            if ( i % 20000 == 0)
            {
                cout << endl << "----------------------------------------" << endl;
                myNet->foo(queue);
                CPU_serial_gng::NeuralGasNetwork* cpuNet = convert_to_cpu_gng(*myNet->gngNetwork);
                if (myNet->gngNetwork->emptyEdgeIndexStack.headIndex < 0 || 
                    myNet->gngNetwork->emptyEdgeIndexStack.headIndex > MAX_STACK_SIZE)
                {
                    cout << "Invalid edge stack. Terminating..." << endl;
                    return;
                }

                cpuNet->writeToFile();
                CPU_serial_gng::LabelGraphNodes(cpuNet);
                delete cpuNet;
                cout << "----------------------------------------" << endl;
            }

            cout << currentDateTime() << "   Iteration " << i << endl;
          //`  getchar();
        }

        float f = getRandomFloat(0,1);
        if (f < 0.75)
        {
            ++i;
            continue;
        }

        vector<float> dataPoint (inputs[i % inputs.size()], inputs[i % inputs.size()] + 784);
        /*
        for (int i = 0; i != dataPoint.size(); ++i)
            cout << dataPoint[i] << " ";
        cout << endl;
        */
        myNet->iterateNetwork(dataPoint, queue);
        ++i;
    }
    myNet->foo(queue);
    CPU_serial_gng::NeuralGasNetwork* cpuNet = convert_to_cpu_gng(*myNet->gngNetwork);

    CPU_serial_gng::LabelGraphNodes(cpuNet);
    cpuNet->writeToFile();
}

void testMNISTFile()
{
    CPU_serial_gng::NeuralGasNetwork* myNet = new CPU_serial_gng::NeuralGasNetwork(
        MNIST_GNG_FILE,
        784,
        NODE_CHANGE_RATE,
        NODE_NEIGHBOR_CHANGE_RATE,
        LOCAL_DECREASE_RATE,
        GLOBAL_DECREASE_RATE,
        AGE_MAX,
        TIME_BETWEEN_ADDING_NODES);

    CPU_serial_gng::LabelGraphNodes(myNet);

    vector<double*> inputs;
    vector<int> targets;
    int inputSize;
    cout << "Reading MNIST data set" << endl;
    readMNIST(inputs, targets, inputSize);
    cout << "Finished reading. Now creating neural net" << endl;
    cout << targets[0];
    cout << endl;

    cout << "ZEROES" << endl;
    for (int i =0; i != 500; ++i)
    {
        vector<double> inputVec (inputs[i % inputs.size()], inputs[i % inputs.size()] + 784);
        int nodeIndex = myNet->findNearest(inputVec);
        if (targets[i] == 0)
            cout << myNet->nodes[nodeIndex]->classification;
    }
    cout << endl << "TWOS" << endl;
    for (int i =0; i != 500; ++i)
    {
        vector<double> inputVec (inputs[i % inputs.size()], inputs[i % inputs.size()] + 784);
        int nodeIndex = myNet->findNearest(inputVec);
        if (targets[i] == 4)
            cout << myNet->nodes[nodeIndex]->classification;
    }
    cout << endl << "SIXES" << endl;
    for (int i =0; i != 500; ++i)
    {
        vector<double> inputVec (inputs[i % inputs.size()], inputs[i % inputs.size()] + 784);
        int nodeIndex = myNet->findNearest(inputVec);
        if (targets[i] == 6)
            cout << myNet->nodes[nodeIndex]->classification;
    }
    cout << endl;


    //Now train the perceptron classifer
    CPU_serial_gng::PerceptronClassifier* myClassifier = new CPU_serial_gng::PerceptronClassifier(myNet, myNet->numberOfClassifications, 10);

    cout << "Training classifiers " << endl;
    for (unsigned int i = 0; i != 10000; ++i)
    {
        vector<double> inputVec (inputs[i % inputs.size()], inputs[i % inputs.size()] + 784);
        myClassifier->train_GNG_Perceptron(inputVec, targets[i % targets.size()]);
    }

    cout << "Finished training " << endl;

    unsigned int numCorrect = 0;
    for (unsigned int i = 0; i != targets.size(); ++i)
    {
        vector<double> inputVec (inputs[i], inputs[i] + 784);
        int result = myClassifier->getOutput(inputVec);
        if (result == targets[i])
            numCorrect += 1;
    }
    cout << "Correct: " << numCorrect << " Total: " << targets.size() << endl;
}
