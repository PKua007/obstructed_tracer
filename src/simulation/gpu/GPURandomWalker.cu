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

namespace {
    class TrajectoriesOnGPU {
    private:
        std::size_t numberOfTrajectories;
        std::size_t numberOfSteps;
        std::vector<Point*> cpuTrajectoryPointers;
        std::vector<std::size_t> cpuAcceptedSteps;

    public:
        Point **gpuTrajectories;
        size_t *gpuAcceptedSteps;

        TrajectoriesOnGPU(std::size_t numberOfTrajectories, std::size_t numberOfSteps) :
                numberOfTrajectories{numberOfTrajectories}, numberOfSteps{numberOfSteps},
                cpuTrajectoryPointers(numberOfTrajectories), cpuAcceptedSteps(numberOfTrajectories)
        { }

        void allocate() {
            cudaCheck( cudaMalloc(&this->gpuTrajectories, this->numberOfTrajectories*sizeof(Point*)) );
            cudaCheck( cudaMalloc(&this->gpuAcceptedSteps, this->numberOfTrajectories*sizeof(size_t)) );

            for (std::size_t i = 0; i < this->numberOfTrajectories; i++)
                cudaCheck( cudaMalloc(&(this->cpuTrajectoryPointers[i]), (this->numberOfSteps + 1) * sizeof(Point)) );
            cudaCheck( cudaMemcpy(this->gpuTrajectories, this->cpuTrajectoryPointers.data(),
                                  this->numberOfTrajectories*sizeof(Point*), cudaMemcpyHostToDevice) );
        }

        void moveToCPU(std::vector<GPUTrajectory> &trajectories) {
            cudaCheck( cudaMemcpy(this->cpuAcceptedSteps.data(), this->gpuAcceptedSteps,
                                  this->numberOfTrajectories*sizeof(size_t), cudaMemcpyDeviceToHost) );

            cudaCheck( cudaFree(this->gpuTrajectories) );
            cudaCheck( cudaFree(this->gpuAcceptedSteps) );

            for (std::size_t i = 0; i < this->numberOfTrajectories; i++) {
                trajectories[i].moveGPUData(this->cpuTrajectoryPointers[i], this->numberOfSteps + 1,
                                            this->cpuAcceptedSteps[i]);
            }
        }
    };

    __global__
    void gpu_random_walk(size_t numberOfTrajectories, size_t numberOfSteps, float tracerRadius, Move drift,
                         MoveGenerator* moveGenerator, MoveFilter* moveFilter, Point **trajectories,
                         size_t *acceptedSteps)
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
    TrajectoriesOnGPU trajectoriesOnGPU(numberOfTrajectories, this->numberOfSteps);
    trajectoriesOnGPU.allocate();

    logger << "[GPURandomWalker::run] Starting simulation... " << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    int numberOfBlocks = (numberOfTrajectories + blockSize - 1) / blockSize;
    gpu_random_walk<<<numberOfBlocks, blockSize>>>(numberOfTrajectories, this->numberOfSteps, this->tracerRadius,
                                                   this->drift, this->moveGenerator, this->moveFilter,
                                                   trajectoriesOnGPU.gpuTrajectories,
                                                   trajectoriesOnGPU.gpuAcceptedSteps);
    cudaCheck( cudaDeviceSynchronize() );
    auto finish = std::chrono::high_resolution_clock::now();
    logger << "completed." << std::endl;

    auto simulationTimeInMus = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    auto singleRunTimeInMus = simulationTimeInMus / this->trajectories.size();
    logger << "[GPURandomWalker::run] Finished after " << simulationTimeInMus << " μs, which gives ";
    logger << singleRunTimeInMus << " μs per trajectory on average." << std::endl;

    trajectoriesOnGPU.moveToCPU(this->trajectories);
}

std::size_t GPURandomWalker::getNumberOfTrajectories() const {
    return this->trajectories.size();
}

const Trajectory &GPURandomWalker::getTrajectory(std::size_t index) const {
    return this->trajectories[index];
}
