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
#include "simulation/Timer.h"


__global__
void gpu_random_walk(size_t numberOfTrajectories, RandomWalker::WalkParameters walkParameters,
                     MoveGenerator* moveGenerator, MoveFilter* moveFilter, Tracer *initialTracers, Point **trajectories,
                     size_t *acceptedSteps)
{
    int i = CUDA_THREAD_IDX;
    if (i >= numberOfTrajectories)
        return;

    Tracer tracer = initialTracers[i];
    trajectories[i][0] = tracer.getPosition();

    acceptedSteps[i] = 0;
    Move rescaledDrift = walkParameters.drift * walkParameters.integrationStep;
    for (size_t step = 1; step <= walkParameters.numberOfSteps; step++) {
        Move move = moveGenerator->generateMove() + rescaledDrift;
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

__global__
void random_valid_tracer_vector(MoveFilter* moveFilter, Tracer *validTracersVector, size_t numberOfTrajectories) {
    int i = CUDA_THREAD_IDX;
    if (i >= numberOfTrajectories)
        return;

    validTracersVector[i] = moveFilter->randomValidTracer();
}

__global__
void delete_objects(MoveGenerator *moveGenerator, MoveFilter *moveFilter) {
    if (!CUDA_IS_IT_FIRST_THREAD)
        return;

    delete moveGenerator;
    delete moveFilter;
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


void GPURandomWalker::TrajectoriesOnGPU::copyToCPU(std::vector<Trajectory> &trajectories) {
    cudaCheck( cudaMemcpy(this->cpuVectorOfAcceptedSteps.data(), this->gpuArrayOfAcceptedSteps,
                          this->numberOfTrajectories*sizeof(size_t), cudaMemcpyDeviceToHost) );

    for (std::size_t i = 0; i < this->numberOfTrajectories; i++) {
        trajectories[i].copyGPUData(this->cpuVectorOfGPUTrajectories[i], this->numberOfSteps + 1,
                                    this->cpuVectorOfAcceptedSteps[i]);
    }
}

GPURandomWalker::GPURandomWalker(std::size_t numberOfTrajectories, RandomWalker::WalkParameters walkParameters,
                                 std::size_t numberOfMoveFilterSetupThreads,  MoveGenerator *moveGenerator,
                                 MoveFilter *moveFilter, std::ostream &logger) :
        numberOfTrajectories{numberOfTrajectories}, walkParameters{walkParameters},
        numberOfMoveFilterSetupThreads{numberOfMoveFilterSetupThreads}, moveGenerator{moveGenerator},
        moveFilter{moveFilter}, trajectoriesOnGPU(numberOfTrajectories, walkParameters.numberOfSteps)
{
    Expects(numberOfTrajectories > 0);
    Expects(walkParameters.numberOfSteps > 0);

    this->trajectories.resize(numberOfTrajectories);
    this->setupMoveFilterForTracerRadius(logger);
}

GPURandomWalker::~GPURandomWalker() {
    delete_objects<<<1, 32>>>(this->moveGenerator, this->moveFilter);
    cudaCheck( cudaDeviceSynchronize() );
}

void GPURandomWalker::setupMoveFilterForTracerRadius(std::ostream& logger) {
    int numberOfBlocks = (this->numberOfMoveFilterSetupThreads + blockSize - 1)
            / blockSize;
    logger << "[GPURandomWalker] Setting up MoveFilter... " << std::flush;
    setup_move_filter<<<numberOfBlocks, blockSize>>>(this->moveFilter, this->walkParameters.tracerRadius);
    cudaCheck(cudaDeviceSynchronize());
    logger << "completed." << std::endl;
}

void GPURandomWalker::printTimerInfo(const Timer &kernelTimer, const Timer &copyTimer, std::ostream &logger) {
    auto kernelTimeInMus = kernelTimer.countMicroseconds();
    auto copyTimeInMus = copyTimer.countMicroseconds();

    auto totalTimeInMus = kernelTimeInMus + copyTimeInMus;
    auto onlyKernelSingleTrajectoryTimeInMus = kernelTimeInMus / this->numberOfTrajectories;
    auto totalSingleTrajectoryTimeInMus = totalTimeInMus / this->numberOfTrajectories;
    logger << "[GPURandomWalker::run] Finished after " << totalTimeInMus << " μs, which gives ";
    logger << onlyKernelSingleTrajectoryTimeInMus << " μs per trajectory on average (";
    logger << totalSingleTrajectoryTimeInMus << " μs with memory fetch)." << std::endl;
}

void GPURandomWalker::run(std::ostream& logger, const std::vector<Tracer> &initialTracers) {
    Tracer *gpuInitialTracers;
    cudaCheck( cudaMalloc(&gpuInitialTracers, this->numberOfTrajectories*sizeof(Tracer)) );
    cudaCheck( cudaMemcpy(gpuInitialTracers, initialTracers.data(), this->numberOfTrajectories*sizeof(Tracer),
                          cudaMemcpyHostToDevice) );

    logger << "[GPURandomWalker::run] Starting simulation... " << std::flush;
    Timer kernelTimer;
    kernelTimer.start();
    int numberOfBlocks = (numberOfTrajectories + blockSize - 1) / blockSize;
    gpu_random_walk<<<numberOfBlocks, blockSize>>>(this->numberOfTrajectories, this->walkParameters,
                                                   this->moveGenerator, this->moveFilter, gpuInitialTracers,
                                                   trajectoriesOnGPU.getTrajectoriesArray(),
                                                   trajectoriesOnGPU.getAcceptedStepsArray());
    cudaCheck( cudaDeviceSynchronize() );
    kernelTimer.stop();
    logger << "completed." << std::endl;

    cudaCheck( cudaFree(gpuInitialTracers) );

    logger << "[GPURandomWalker::run] Fetching data from video memory... " << std::flush;
    Timer copyTimer;
    copyTimer.start();
    trajectoriesOnGPU.copyToCPU(this->trajectories);
    copyTimer.stop();
    logger << "completed." << std::endl;

    this->printTimerInfo(kernelTimer, copyTimer, logger);
}

std::size_t GPURandomWalker::getNumberOfTrajectories() const {
    return this->numberOfTrajectories;
}

std::size_t GPURandomWalker::getNumberOfSteps() const {
    return this->walkParameters.numberOfSteps;
}

std::vector<Tracer> GPURandomWalker::getRandomInitialTracersVector() {
    std::vector<Tracer> cpuInitialTracers(this->numberOfTrajectories);

    Tracer *gpuInitialTracers;
    cudaCheck( cudaMalloc(&gpuInitialTracers, this->numberOfTrajectories*sizeof(Tracer)) );
    int numberOfBlocks = (numberOfTrajectories + blockSize - 1) / blockSize;
    random_valid_tracer_vector<<<numberOfBlocks, blockSize>>>(this->moveFilter, gpuInitialTracers,
                                                              this->numberOfTrajectories);
    cudaCheck( cudaMemcpy(cpuInitialTracers.data(), gpuInitialTracers, this->numberOfTrajectories*sizeof(Tracer),
                          cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(gpuInitialTracers) );

    return cpuInitialTracers;
}

const Trajectory &GPURandomWalker::getTrajectory(std::size_t index) const {
    return this->trajectories[index];
}

const std::vector<Trajectory> &GPURandomWalker::getTrajectories() const {
    return this->trajectories;
}
