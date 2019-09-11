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
#include "simulation/SimulationTimer.h"


__global__
void gpu_random_walk(size_t numberOfTrajectories, RandomWalker::WalkParameters walkParameters,
                     MoveGenerator* moveGenerator, MoveFilter* moveFilter, Point **trajectories,
                     size_t *acceptedSteps)
{
    int i = CUDA_THREAD_IDX;
    if (i >= numberOfTrajectories)
        return;

    Tracer tracer = moveFilter->randomValidTracer();
    trajectories[i][0] = tracer.getPosition();

    acceptedSteps[i] = 0;
    for (size_t step = 1; step <= walkParameters.numberOfSteps; step++) {
        Move move = moveGenerator->generateMove() + walkParameters.drift;
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

GPURandomWalker::TrajectoriesOnGPU::TrajectoriesOnGPU(std::size_t numberOfTrajectories, std::size_t numberOfSteps) :
        numberOfTrajectories{numberOfTrajectories}, numberOfSteps{numberOfSteps},
        cpuVectorOfGPUTrajectories(numberOfTrajectories), cpuVectorOfAcceptedSteps(numberOfTrajectories)
{
    cudaCheck( cudaMalloc(&this->gpuArrayOfGPUTrajectories, this->numberOfTrajectories*sizeof(Point*)) );
    cudaCheck( cudaMalloc(&this->gpuArrayOfAcceptedSteps, this->numberOfTrajectories*sizeof(size_t)) );

    for (std::size_t i = 0; i < this->numberOfTrajectories; i++) {
        // Number of steps plus ONE STEP for initial tracer
        cudaCheck( cudaMalloc(&(this->cpuVectorOfGPUTrajectories[i]),
                              (this->numberOfSteps + 1) * sizeof(Point)) );
    }
    cudaCheck( cudaMemcpy(this->gpuArrayOfGPUTrajectories, this->cpuVectorOfGPUTrajectories.data(),
                          this->numberOfTrajectories*sizeof(Point*), cudaMemcpyHostToDevice) );
}

GPURandomWalker::TrajectoriesOnGPU::~TrajectoriesOnGPU() {
    cudaCheck( cudaFree(this->gpuArrayOfGPUTrajectories) );
    for (auto gpuTrajectory : cpuVectorOfGPUTrajectories)
        cudaCheck( cudaFree(gpuTrajectory) );
    cudaCheck( cudaFree(this->gpuArrayOfAcceptedSteps) );
}


void GPURandomWalker::TrajectoriesOnGPU::copyToCPU(std::vector<GPUTrajectory> &trajectories) {
    cudaCheck( cudaMemcpy(this->cpuVectorOfAcceptedSteps.data(), this->gpuArrayOfAcceptedSteps,
                          this->numberOfTrajectories*sizeof(size_t), cudaMemcpyDeviceToHost) );

    for (std::size_t i = 0; i < this->numberOfTrajectories; i++) {
        trajectories[i].copyGPUData(this->cpuVectorOfGPUTrajectories[i], this->numberOfSteps + 1,
                                    this->cpuVectorOfAcceptedSteps[i]);
    }
}

GPURandomWalker::GPURandomWalker(std::size_t numberOfTrajectories, RandomWalker::WalkParameters walkParameters,
                                 std::size_t numberOfMoveFilterSetupThreads,  MoveGenerator* moveGenerator,
                                 MoveFilter* moveFilter, std::ostream &logger) :
        numberOfTrajectories{numberOfTrajectories}, walkParameters{walkParameters},
        numberOfMoveFilterSetupThreads{numberOfMoveFilterSetupThreads}, moveGenerator{moveGenerator},
        moveFilter{moveFilter}, trajectoriesOnGPU(numberOfTrajectories, walkParameters.numberOfSteps)
{
    Expects(numberOfTrajectories > 0);
    Expects(numberOfSteps > 0);
    Expects(tracerRadius >= 0.f);

    this->trajectories.resize(numberOfTrajectories);
    this->setupMoveFilterForTracerRadius(logger);
}

void GPURandomWalker::setupMoveFilterForTracerRadius(std::ostream& logger) {
    int numberOfBlocks = (this->numberOfMoveFilterSetupThreads + blockSize - 1)
            / blockSize;
    logger << "[GPURandomWalker::run] Setting up MoveFilter... " << std::flush;
    setup_move_filter<<<numberOfBlocks, blockSize>>>(this->moveFilter, this->walkParameters.tracerRadius);
    cudaCheck(cudaDeviceSynchronize());
    logger << "completed." << std::endl;
}

void GPURandomWalker::run(std::ostream& logger) {
    SimulationTimer timer(this->numberOfTrajectories);
    timer.start();

    logger << "[GPURandomWalker::run] Starting simulation... " << std::flush;
    int numberOfBlocks = (numberOfTrajectories + blockSize - 1) / blockSize;
    gpu_random_walk<<<numberOfBlocks, blockSize>>>(this->numberOfTrajectories, this->walkParameters,
                                                   this->moveGenerator, this->moveFilter,
                                                   trajectoriesOnGPU.getTrajectoriesArray(),
                                                   trajectoriesOnGPU.getAcceptedStepsArray());
    cudaCheck( cudaDeviceSynchronize() );
    logger << "completed." << std::endl;

    logger << "[GPURandomWalker::run] Fetching data from video memory... " << std::flush;
    trajectoriesOnGPU.copyToCPU(this->trajectories);
    logger << "completed." << std::endl;

    timer.stop();
    timer.showInfo(logger);
}

std::size_t GPURandomWalker::getNumberOfTrajectories() const {
    return this->numberOfTrajectories;
}

const Trajectory &GPURandomWalker::getTrajectory(std::size_t index) const {
    return this->trajectories[index];
}
