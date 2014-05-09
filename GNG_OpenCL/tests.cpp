#include "test.h"

//#define GNG_NN_TEST_FILE            "GNG-NN.net"

vector<float> generateTestData()
{
    //Generates a data point either in the range (0.75,0.75) -> (1,1) or (0,0) -> (.25,.25)
    double g = getRandomFloat(0, 1);
    double lb;
    double ub;
    if (g < 0.33)
    {
        lb = 0.8;
        ub = 1.00;
    }
    else if (g < 0.66)
    {
        lb = 0.0;
        ub = 0.2;
    }
    else
    {
        lb = 0.4;
        ub = 0.6;
    }
    vector<float> generatedVector;
    generatedVector.push_back(getRandomFloat(lb, ub));
    generatedVector.push_back(getRandomFloat(lb, ub));
    generatedVector.shrink_to_fit();
    return generatedVector;
}

void runTest()
{

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
        2,
        NODE_CHANGE_RATE,
        NODE_NEIGHBOR_CHANGE_RATE,
        LOCAL_DECREASE_RATE,
        GLOBAL_DECREASE_RATE,
        AGE_MAX,
        TIME_BETWEEN_ADDING_NODES,
        context,
        opencl_gngNetworkProgram,
        queue);

    for (int i = 0; i != 50000; ++i)
    {
        if (i%500 == 0)
            cout << "iteration: " << i << endl;
        vector<float> dataPoint = generateTestData();
        myNet->iterateNetwork(dataPoint, queue);
    }
    myNet->foo(queue);
    CPU_serial_gng::NeuralGasNetwork* cpuNet = convert_to_cpu_gng(*myNet->gngNetwork);

    CPU_serial_gng::LabelGraphNodes(cpuNet);

}