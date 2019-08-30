/*
 * GPUTrajectory.cpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include <ostream>

#include "GPUTrajectory.h"
#include "utils/CudaCheck.h"

void GPUTrajectory::moveGPUData(Point* gpuData, std::size_t size, std::size_t acceptedSteps) {
    this->acceptedSteps = acceptedSteps;
    this->trajectory.resize(size);
    cudaCheck( cudaMemcpy(this->trajectory.data(), gpuData, size*sizeof(Point), cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(gpuData) );
}
