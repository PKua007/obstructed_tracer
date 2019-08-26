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

std::size_t GPUTrajectory::getSize() const {
    return this->trajectory.size();
}

std::size_t GPUTrajectory::getNumberOfAcceptedSteps() const {
    return this->acceptedSteps;
}

Point GPUTrajectory::operator[](std::size_t index) const {
    return this->trajectory[index];
}

Point GPUTrajectory::getFirst() const {
    return this->trajectory.front();
}

Point GPUTrajectory::getLast() const {
    return this->trajectory.back();
}

void GPUTrajectory::store(std::ostream& out) const {
    for (auto point : this->trajectory)
        out << point.x << " " << point.y << "\n";
}
