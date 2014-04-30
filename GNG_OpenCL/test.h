#ifndef TEST_H
#define TEST_H

#include "general_include.h"
#include "utils.h"
#include "gpu_gng.h"
#include "graph_algorithms.h"
#include "convert_cpu_gpu_gngs.h"

//Generates some test data
vector<float> generateTestData();

//Runs some tests on the test data
void runTest();

#endif
