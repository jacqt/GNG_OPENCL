#ifndef CONVERT_CPU_GPU_H
#define CONVERT_CPU_GPU_H

#include "general_include.h"
#include "utils.h"
#include "cpu_gng.h"
#include "gpu_gng.h"

CPU_serial_gng::NeuralGasNetwork* convert_to_cpu_gng(GPU_parallel_gng::NeuralGasNetwork &originalNet);

#endif