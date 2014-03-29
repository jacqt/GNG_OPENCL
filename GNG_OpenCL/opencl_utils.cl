#ifndef OPENCL_UTILS_H
#define OPENCL_UTILS_H

#define MAX_STACK_SIZE          2000

typedef struct IntStack{
    int headIndex;
    int stack[MAX_STACK_SIZE];
} IntStack;

//Pre: stackSize < MAX_STACK_SIZE
//Creates and returns an IntStack with the values 0,1, 2, 3, ..., (stacksize-1)

bool IntStack_isEmpty(global IntStack* intStack)
{
    if (intStack->headIndex == -1)
        return true;
    return false;
}

int IntStack_pop(global IntStack* intStack)
{
    --intStack->headIndex;
    return intStack->stack[intStack->headIndex + 1];
}

void IntStack_push(global IntStack* intStack, int n)
{
    ++intStack->headIndex;
    intStack->stack[intStack->headIndex] = n;
}
float getRandomFloat(float lowerBound, float upperBound);

#else
#endif