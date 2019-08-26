/*
 * GPUSimulationFactory.cpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include "GPUSimulationFactory.h"
#include "utils/CudaCheck.h"
#include "move_generator/GPUGaussianMoveGenerator.h"
#include "move_filter/DefaultMoveFilter.h"

__global__
void create_move_generator_and_filter(unsigned long seed, float sigma, size_t numberOfTrajectories,
                                      MoveGenerator **moveGenerator, MoveFilter **moveFilter)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i != 0)
        return;

    (*moveGenerator) = new GPUGaussianMoveGenerator(sigma, seed, numberOfTrajectories);
    (*moveFilter) = new DefaultMoveFilter();
}

__global__
void delete_move_generator_and_filter(MoveGenerator *moveGenerator, MoveFilter *moveFilter)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i != 0)
        return;

    delete moveGenerator;
    delete moveFilter;
}

GPUSimulationFactory::GPUSimulationFactory(const Parameters& parameters, std::ostream& logger) {
    if (parameters.seed == "random") {
        unsigned long randomSeed = std::random_device()();
        this->seedGenerator.seed(randomSeed);
        logger << "[GPUSimulationFactory] Using random seed: " << randomSeed << std::endl;
    } else {
        this->seedGenerator.seed(std::stoul(parameters.seed));
    }

    MoveGenerator **moveGeneratorPlaceholder;
    MoveFilter **moveFilterPlaceholder;
    cudaCheck( cudaMalloc(&moveGeneratorPlaceholder, sizeof(MoveGenerator**)) );
    cudaCheck( cudaMalloc(&moveFilterPlaceholder, sizeof(MoveFilter**)) );

    create_move_generator_and_filter<<<1, 32>>>(this->seedGenerator(), 2.f, parameters.numberOfWalks,
                                                moveGeneratorPlaceholder, moveFilterPlaceholder);
    cudaCheck( cudaPeekAtLastError() );

    cudaCheck( cudaMemcpy(&(this->moveGenerator), moveGeneratorPlaceholder, sizeof(MoveGenerator*),
                          cudaMemcpyDeviceToHost) );
    cudaCheck( cudaMemcpy(&(this->moveFilter), moveFilterPlaceholder, sizeof(MoveFilter*),
                          cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(moveGeneratorPlaceholder) );
    cudaCheck( cudaFree(moveFilterPlaceholder) );

    Move drift = {parameters.driftX, parameters.driftY};
    this->randomWalker.reset(new GPURandomWalker(parameters.numberOfWalks, parameters.numberOfSteps,
                                                 parameters.tracerRadius, drift, this->moveGenerator,
                                                 this->moveFilter));
}

GPUSimulationFactory::~GPUSimulationFactory() {
    delete_move_generator_and_filter<<<1, 32>>>(this->moveGenerator, this->moveFilter);
    cudaCheck( cudaPeekAtLastError() );
}

RandomWalker& GPUSimulationFactory::getRandomWalker() {
    return *this->randomWalker;
}
