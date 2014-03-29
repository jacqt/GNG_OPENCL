#ifndef UTILS_H
#define UTILS_H

#define MAX_STACK_SIZE          2000

#include "general_include.h"

typedef struct IntStack{
    int headIndex;
    int stack[MAX_STACK_SIZE];
} IntStack;

//Pre: stackSize < MAX_STACK_SIZE
//Creates and returns an IntStack with the values 0,1, 2, 3, ..., (stacksize-1)
IntStack createNewStack(unsigned int stackSize);

bool IntStack_isEmpty(IntStack intStack);
int  IntStack_pop(IntStack* intStack);
void  IntStack_push(IntStack* intStack, int n);

float getRandomFloat(float lowerbound, float upperbound);

cl::Program createProgram(cl::Context &context, string fname, const string params = "");
string getFileContents(const char* fileName);

#endif