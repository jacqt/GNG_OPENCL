#include "gng_network.h"

int main()
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

    return 0;
}