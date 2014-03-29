#include "utils.h"

IntStack createNewStack(unsigned int stackSize)
{
    IntStack* intStack = new IntStack;
    for (unsigned int i = 0; i != stackSize; ++i)
        intStack->stack[i] = stackSize - i - 1;
    intStack->headIndex = stackSize-1;
    return (*intStack);
}

bool IntStack_isEmpty(IntStack* intStack)
{
    if (intStack->headIndex == -1)
        return true;
    return false;
}

int IntStack_pop(IntStack* intStack)
{
    --intStack->headIndex;
    return intStack->stack[intStack->headIndex + 1];
}

void IntStack_push(IntStack* intStack, int n)
{
    ++intStack->headIndex;
    intStack->stack[intStack->headIndex] = n;
}

float getRandomFloat(float lowerbound, float upperbound)
{
	float f = (float) rand() / RAND_MAX;
	f = lowerbound + f * (upperbound - lowerbound);
	return f;
}


cl::Program createProgram(cl::Context &context, std::string fname, const std::string params) 
{
    cl::Program::Sources sources;
    vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES> ();
    
    std::string sourceCode = getFileContents(fname.c_str());
    sources.insert(sources.end(), std::make_pair(sourceCode.c_str(),
        sourceCode.length()));
    cl::Program* program = new cl::Program(context,sources);

    try
    {
        (*program).build(devices, params.c_str());
    }
    catch (cl::Error e)
    {
        cout << "Compilation build error log: " << endl <<
            (*program).getBuildInfo <CL_PROGRAM_BUILD_LOG> (devices [0]) << endl;
    }

    return (*program);
}

string getFileContents(const char* fileName)
{
    std::ifstream in(fileName, std::ios::in | std::ios::binary);
    if (in)
        return(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
    throw(errno);
}
