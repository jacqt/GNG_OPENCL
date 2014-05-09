#ifndef GENERAL_INCLUDE_H
#define GENERAL_INCLUDE_H

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>
#include <time.h>
#include <climits>



using std::cout;
using std::cin;
using std::endl;
using std::vector;
using std::string;


#define NODE_CHANGE_RATE                0.05
#define NODE_NEIGHBOR_CHANGE_RATE       0.00005
#define LOCAL_DECREASE_RATE             0.5
#define GLOBAL_DECREASE_RATE            0.9995
#define AGE_MAX                         10000

//Stuff above is irrelevant
#define TIME_BETWEEN_ADDING_NODES       300
#define MAX_NODES                       8192
#define MAX_EDGES                       16384
#define MAX_DIMENSION                   1024

#define MAX_STACK_SIZE                  17000

#endif