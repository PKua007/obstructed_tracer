/*
 * GPURandomWalker.cpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include <stdexcept>
#include <ostream>

#include "GPURandomWalker.h"
#include "utils/Assertions.h"
#include "utils/CudaCheck.h"

__global__
void gpu_random_walk(size_t numberOfTrajectories, size_t numberOfSteps, float tracerRadius, Move drift,
                     MoveGenerator* moveGenerator, MoveFilter* moveFilter, Point **trajectories, size_t *acceptedSteps)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= numberOfTrajectories)
        return;

    Tracer tracer = moveFilter->randomValidTracer(tracerRadius);
    trajectories[i][0] = tracer.getPosition();

    acceptedSteps[i] = 0;
    for (size_t step = 1; step <= numberOfSteps; step++) {
        Move move = moveGenerator->generateMove() + drift;
        if (moveFilter->isMoveValid(tracer, move)) {
            tracer += move;
            trajectories[i][step] = tracer.getPosition();
            acceptedSteps[i]++;
        } else {
            trajectories[i][step] = tracer.getPosition();
        }
    }
}

GPURandomWalker::GPURandomWalker(std::size_t numberOfTrajectories, std::size_t numberOfSteps, float tracerRadius,
                                 Move drift, MoveGenerator* moveGenerator, MoveFilter* moveFilter) :
        numberOfSteps{numberOfSteps}, tracerRadius{tracerRadius}, drift{drift}, moveGenerator{moveGenerator},
        moveFilter{moveFilter} {
    Expects(numberOfTrajectories > 0);
    Expects(numberOfSteps > 0);
    Expects(tracerRadius >= 0.f);
    this->trajectories.resize(numberOfTrajectories);
}

void GPURandomWalker::run(std::ostream& logger) {
    std::size_t numberOfTrajectories = this->trajectories.size();

    Point **gpuTrajectories;
    size_t *gpuAcceptedSteps;
    cudaCheck( cudaMalloc(&gpuTrajectories, numberOfTrajectories*sizeof(Point*)) );
    cudaCheck( cudaMalloc(&gpuAcceptedSteps, numberOfTrajectories*sizeof(size_t)) );

    std::vector<Point*> cpuTrajectoryPointers(numberOfTrajectories);
    for (std::size_t i = 0; i < numberOfTrajectories; i++)
        cudaCheck( cudaMalloc(&(cpuTrajectoryPointers[i]), (this->numberOfSteps + 1) * sizeof(Point)) );
    cudaCheck( cudaMemcpy(gpuTrajectories, cpuTrajectoryPointers.data(), numberOfTrajectories*sizeof(Point*),
                          cudaMemcpyHostToDevice) );

    logger << "[GPURandomWalker::run] Starting simulation... " << std::flush;

    int blockSize = 32;
    int numberOfBlocks = (numberOfTrajectories + blockSize - 1) / blockSize;
    gpu_random_walk<<<numberOfBlocks, blockSize>>>(numberOfTrajectories, this->numberOfSteps, this->tracerRadius,
                                                   this->drift, this->moveGenerator, this->moveFilter, gpuTrajectories,
                                                   gpuAcceptedSteps);

    cudaCheck( cudaPeekAtLastError() );

    logger << "completed." << std::endl;

    cudaCheck( cudaFree(gpuTrajectories) );

    std::vector<size_t> cpuAcceptedSteps(numberOfTrajectories);
    cudaCheck( cudaMemcpy(cpuAcceptedSteps.data(), gpuAcceptedSteps, numberOfTrajectories*sizeof(size_t),
                          cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(gpuAcceptedSteps) );

    for (std::size_t i = 0; i < numberOfTrajectories; i++)
        this->trajectories[i].moveGPUData(cpuTrajectoryPointers[i], (this->numberOfSteps + 1), cpuAcceptedSteps[i]);
}

std::size_t GPURandomWalker::getNumberOfTrajectories() const {
    return this->trajectories.size();
}

const Trajectory &GPURandomWalker::getTrajectory(std::size_t index) const {
    return this->trajectories[index];
}
