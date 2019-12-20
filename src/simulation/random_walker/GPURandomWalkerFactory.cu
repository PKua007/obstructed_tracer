/*
 * GPURandomWalkerFactory.cpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include <sstream>
#include <fstream>
#include <vector>

#include "GPURandomWalkerFactory.h"
#include "utils/CudaCheck.h"
#include "simulation/move_generator/gpu/GPUGaussianMoveGenerator.h"
#include "simulation/move_generator/gpu/GPUCauchyMoveGenerator.h"
#include "simulation/move_filter/DefaultMoveFilter.h"
#include "simulation/move_filter/image_move_filter/ImageMoveFilter.h"
#include "simulation/move_filter/image_move_filter/WallBoundaryConditions.h"
#include "simulation/move_filter/image_move_filter/PeriodicBoundaryConditions.h"
#include "image/PPMImageReader.h"
#include "utils/Assertions.h"


__global__
void create_move_generator(unsigned long seed, float sigma, float integrationStep, size_t numberOfTrajectories,
                           GPURandomWalkerFactory::MoveGeneratorType moveGeneratorType, MoveGenerator **moveGenerator)
{
    if (!CUDA_IS_IT_FIRST_THREAD)
        return;

    using MoveGeneratorType = GPURandomWalkerFactory::MoveGeneratorType;

    if (moveGeneratorType == MoveGeneratorType::GAUSSIAN)
        (*moveGenerator) = new GPUGaussianMoveGenerator(sigma, integrationStep, seed, numberOfTrajectories);
    else if (moveGeneratorType == MoveGeneratorType::CAUCHY)
        (*moveGenerator) = new GPUCauchyMoveGenerator(sigma, integrationStep, seed, numberOfTrajectories);
    else
        (*moveGenerator) = nullptr;
}


__global__
void create_move_filter(unsigned long seed, size_t numberOfTrajectories,
                        GPURandomWalkerFactory::MoveFilterType moveFilterType, uint32_t *intImageData, size_t width,
                        size_t height, GPURandomWalkerFactory::BoundaryConditionsType boundaryConditionsType,
                        MoveFilter **moveFilter)
{
    if (!CUDA_IS_IT_FIRST_THREAD)
        return;

    using MoveFilterType = GPURandomWalkerFactory::MoveFilterType;
    using BoundaryConditionsType = GPURandomWalkerFactory::BoundaryConditionsType;

    if (moveFilterType == MoveFilterType::DEFAULT) {
        (*moveFilter) = new DefaultMoveFilter();
    } else if (moveFilterType == MoveFilterType::IMAGE) {
        if (boundaryConditionsType == BoundaryConditionsType::WALL) {
            (*moveFilter) = new ImageMoveFilter<WallBoundaryConditions>(intImageData, width, height, seed,
                                                                        numberOfTrajectories);
        } else if (boundaryConditionsType == BoundaryConditionsType::PERIODIC) {
            (*moveFilter) = new ImageMoveFilter<PeriodicBoundaryConditions>(intImageData, width, height, seed,
                                                                            numberOfTrajectories);
        } else {
            (*moveFilter) = nullptr;
        }
    } else {
        (*moveFilter) = nullptr;
    }
}

GPURandomWalkerFactory::MoveGeneratorOnGPUFactory::MoveGeneratorOnGPUFactory(const std::string &moveGeneratorString,
                                                                             float integrationStep)
        : integrationStep{integrationStep}
{
    Validate(integrationStep > 0.f);

    std::istringstream moveGeneratorStream(moveGeneratorString);
    std::string moveGeneratorName;
    moveGeneratorStream >> moveGeneratorName >> this->sigma;
    if (!moveGeneratorStream)
        throw std::runtime_error("Malformed MoveGenerator parameters");
    Validate(this->sigma >= 0.f);

    if (moveGeneratorName == "GaussianMoveGenerator")
        this->moveGeneratorType = GAUSSIAN;
    else if (moveGeneratorName == "CauchyMoveGenerator")
        this->moveGeneratorType =  CAUCHY;
    else
        throw std::runtime_error("Unknown MoveGenerator: " + moveGeneratorName);
}

MoveGenerator *GPURandomWalkerFactory::MoveGeneratorOnGPUFactory::create(unsigned long seed,
                                                                         std::size_t numberOfWalks)
{
    MoveGenerator **moveGeneratorPlaceholder{};
    cudaCheck( cudaMalloc(&moveGeneratorPlaceholder, sizeof(MoveGenerator**)) );
    create_move_generator<<<1, 32>>>(seed, this->sigma, this->integrationStep, numberOfWalks, this->moveGeneratorType,
                                     moveGeneratorPlaceholder);
    cudaCheck( cudaDeviceSynchronize() );

    MoveGenerator *moveGenerator;
    cudaCheck( cudaMemcpy(&moveGenerator, moveGeneratorPlaceholder, sizeof(MoveGenerator*),
                          cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(moveGeneratorPlaceholder) );

    return moveGenerator;
}

void GPURandomWalkerFactory::MoveFilterOnGPUFactory::fetchImageData(std::istringstream &moveFilterStream,
                                                                    std::ostream &logger)
{
    std::string imageFilename;
    moveFilterStream >> imageFilename;
    if (!moveFilterStream)
        throw std::runtime_error("Malformed ImageMoveFilter parameters");

    std::ifstream imageFile(imageFilename);
    if (!imageFile)
        throw std::runtime_error("Cannot open " + imageFilename + " to load image");

    PPMImageReader imageReader;
    this->image = imageReader.read(imageFile);
    logger << "[GPURandomWalkerFactory] Loaded image " << imageFilename << " (" << this->image.getWidth();
    logger << "px x " << this->image.getHeight() << "px)" << std::endl;
}

void GPURandomWalkerFactory::MoveFilterOnGPUFactory::fetchBoundaryConditions(std::istringstream &moveFilterStream) {
    std::string imageBCType;
    moveFilterStream >> imageBCType;
    if (!moveFilterStream)
        throw std::runtime_error("Malformed ImageMoveFilter parameters");

    if (imageBCType == "WallBoundaryConditions")
        this->boundaryConditionsType = WALL;
    else if (imageBCType == "PeriodicBoundaryConditions")
        this->boundaryConditionsType = PERIODIC;
    else
        throw std::runtime_error("Unknown ImageBoundaryConditions: " + imageBCType);
}

GPURandomWalkerFactory::MoveFilterOnGPUFactory::MoveFilterOnGPUFactory(const std::string &moveFilterString,
                                                                       std::ostream &logger)
{
    std::istringstream moveFilterStream(moveFilterString);
    std::string moveFilterName;
    moveFilterStream >> moveFilterName;

    if (moveFilterName == "DefaultMoveFilter")
        this->moveFilterType = DEFAULT;
    else if (moveFilterName == "ImageMoveFilter")
        this->moveFilterType = IMAGE;
    else
        throw std::runtime_error("Unknown MoveFilter: " + moveFilterName);

    if (this->moveFilterType == IMAGE) {
        this->fetchImageData(moveFilterStream, logger);
        this->fetchBoundaryConditions(moveFilterStream);
        this->numberOfSetupThreads = this->image.getNumberOfPixels();
    } else {    // this->moveFilterType == DEFAULT
        this->numberOfSetupThreads = 1;
    }
}

MoveFilter *GPURandomWalkerFactory::MoveFilterOnGPUFactory::create(unsigned long seed, std::size_t numberOfWalks) {
    MoveFilter **moveFilterPlaceholder{};
    uint32_t *gpuIntImageData{};

    cudaCheck( cudaMalloc(&moveFilterPlaceholder, sizeof(MoveFilter**)) );

    auto intImageData = this->image.getIntData();
    if (this->moveFilterType == IMAGE) {
        cudaCheck( cudaMalloc(&gpuIntImageData, intImageData.size()*sizeof(uint32_t)));
        cudaCheck( cudaMemcpy(gpuIntImageData, intImageData.data(), intImageData.size()*sizeof(uint32_t),
                              cudaMemcpyHostToDevice) );
    }

    create_move_filter<<<1, 32>>>(seed, numberOfWalks, this->moveFilterType, gpuIntImageData,
                                  this->image.getWidth(), this->image.getHeight(), this->boundaryConditionsType,
                                  moveFilterPlaceholder);
    cudaCheck( cudaDeviceSynchronize() );

    MoveFilter *moveFilter;
    cudaCheck( cudaMemcpy(&moveFilter, moveFilterPlaceholder, sizeof(MoveFilter*), cudaMemcpyDeviceToHost) );

    cudaCheck( cudaFree(moveFilterPlaceholder) );
    cudaCheck( cudaFree(gpuIntImageData) );

    return moveFilter;
}


GPURandomWalkerFactory::GPURandomWalkerFactory(unsigned long seed,
                                               const RandomWalkerFactory::WalkerParameters &walkerParameters,
                                               std::ostream &logger)
        : walkerParameters{walkerParameters}, numberOfWalksInSeries{walkerParameters.numberOfWalksInSeries},
          logger{logger},
          gpuMoveGeneratorFactory(walkerParameters.moveGeneratorParameters,
                                  walkerParameters.walkParameters.integrationStep),
          gpuMoveFilterFactory(walkerParameters.moveFilterParameters, logger), seedGenerator(seed)
{ }

std::unique_ptr<RandomWalker> GPURandomWalkerFactory::createRandomWalker() {
    MoveGenerator *moveGenerator = this->gpuMoveGeneratorFactory.create(this->seedGenerator(),
                                                                        this->numberOfWalksInSeries);
    MoveFilter *moveFilter = this->gpuMoveFilterFactory.create(this->seedGenerator(), this->numberOfWalksInSeries);

    return std::unique_ptr<RandomWalker>(
        new GPURandomWalker(this->numberOfWalksInSeries, walkerParameters.walkParameters,
                            gpuMoveFilterFactory.numberOfSetupThreads, moveGenerator, moveFilter, logger)
    );
}
