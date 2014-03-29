#ifndef GENERAL_INCLUDE_H
#define GENERAL_INCLUDE_H

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <fstream>


using std::cout;
using std::cin;
using std::endl;
using std::vector;
using std::string;


#define NODE_CHANGE_RATE                0.05
#define NODE_NEIGHBOR_CHANGE_RATE       0.00005
#define LOCAL_DECREASE_RATE             0.5
#define GLOBAL_DECREASE_RATE            0.9995
#define AGE_MAX                         3200
#define TIME_BETWEEN_ADDING_NODES       300
#define MAX_NODES                       1000
#define MAX_EDGES                       1000
#define MAX_DIMENSION                   200

#endif