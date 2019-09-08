/*
 * GPURandomWalker.cpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include <stdexcept>
#include <ostream>
#include <chrono>

#include "GPURandomWalker.h"
#include "utils/Assertions.h"
#include "utils/CudaCheck.h"

__global__
void gpu_random_walk(size_t numberOfTrajectories, size_t numberOfSteps, float tracerRadius, Move drift,
                     MoveGenerator* moveGenerator, MoveFilter* moveFilter, Point **trajectories, size_t *acceptedSteps)
{
    int i = CUDA_THREAD_IDX;
    if (i >= numberOfTrajectories)
        return;

    Tracer tracer = moveFilter->randomValidTracer();
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

__global__
void setup_move_filter(MoveFilter* moveFilter, float tracerRadius) {
    moveFilter->setupForTracerRadius(tracerRadius);
}

GPURandomWalker::GPURandomWalker(std::size_t numberOfTrajectories, std::size_t numberOfSteps,
                                 std::size_t numberOfMoveFilterSetupThreads, float tracerRadius, Move drift,
                                 MoveGenerator* moveGenerator, MoveFilter* moveFilter) :
        numberOfSteps{numberOfSteps}, numberOfMoveFilterSetupThreads{numberOfMoveFilterSetupThreads},
        tracerRadius{tracerRadius}, drift{drift}, moveGenerator{moveGenerator}, moveFilter{moveFilter}
{
    Expects(numberOfTrajectories > 0);
    Expects(numberOfSteps > 0);
    Expects(tracerRadius >= 0.f);
    this->trajectories.resize(numberOfTrajectories);
}

void GPURandomWalker::setupMoveFilterForTracerRadius(std::ostream& logger) {
    int numberOfBlocks = (this->numberOfMoveFilterSetupThreads + blockSize - 1)
            / blockSize;
    logger << "[GPURandomWalker::run] Setting up MoveFilter... " << std::flush;
    setup_move_filter<<<numberOfBlocks, blockSize>>>(this->moveFilter, this->tracerRadius);
    cudaCheck(cudaDeviceSynchronize());
    logger << "completed." << std::endl;
}

void GPURandomWalker::run(std::ostream& logger) {
    this->setupMoveFilterForTracerRadius(logger);

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

    auto start = std::chrono::high_resolution_clock::now();
    int numberOfBlocks = (numberOfTrajectories + blockSize - 1) / blockSize;
    gpu_random_walk<<<numberOfBlocks, blockSize>>>(numberOfTrajectories, this->numberOfSteps, this->tracerRadius,
                                                   this->drift, this->moveGenerator, this->moveFilter, gpuTrajectories,
                                                   gpuAcceptedSteps);
    cudaCheck( cudaDeviceSynchronize() );
    auto finish = std::chrono::high_resolution_clock::now();
    logger << "completed." << std::endl;

    auto simulationTimeInMus = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    auto singleRunTimeInMus = simulationTimeInMus / this->trajectories.size();
    logger << "[GPURandomWalker::run] Finished after " << simulationTimeInMus << " μs, which gives ";
    logger << singleRunTimeInMus << " μs per trajectory on average." << std::endl;

    std::vector<size_t> cpuAcceptedSteps(numberOfTrajectories);
    cudaCheck( cudaMemcpy(cpuAcceptedSteps.data(), gpuAcceptedSteps, numberOfTrajectories*sizeof(size_t),
                          cudaMemcpyDeviceToHost) );

    cudaCheck( cudaFree(gpuTrajectories) );
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
