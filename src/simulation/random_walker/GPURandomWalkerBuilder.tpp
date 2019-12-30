/*
 * GPURandomWalkerBuilder.tpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include <sstream>
#include <fstream>
#include <vector>

#include "utils/CudaCheck.h"
#include "image/PPMImageReader.h"
#include "utils/Assertions.h"


template<typename GPURandomWalker_t>
__global__
void create_move_generator(unsigned long seed, float sigma, float integrationStep, size_t numberOfTrajectories,
                           typename GPURandomWalkerBuilder<GPURandomWalker_t>::MoveGeneratorType moveGeneratorType,
                           MoveGenerator **moveGenerator)
{
    if (!CUDA_IS_IT_FIRST_THREAD)
        return;

    using GPURandomWalkerBuilder_t = GPURandomWalkerBuilder<GPURandomWalker_t>;
    using MoveGeneratorType = typename GPURandomWalkerBuilder_t::MoveGeneratorType;
    using GaussianMoveGenerator = typename GPURandomWalkerBuilder_t::GaussianMoveGenerator_t;
    using CauchyMoveGenerator = typename GPURandomWalkerBuilder_t::CauchyMoveGenerator_t;

    if (moveGeneratorType == MoveGeneratorType::GAUSSIAN)
        (*moveGenerator) = new GaussianMoveGenerator(sigma, integrationStep, seed, numberOfTrajectories);
    else if (moveGeneratorType == MoveGeneratorType::CAUCHY)
        (*moveGenerator) = new CauchyMoveGenerator(sigma, integrationStep, seed, numberOfTrajectories);
    else
        (*moveGenerator) = nullptr;
}

template<typename GPURandomWalker_t>
__global__
void create_move_filter(unsigned long seed, size_t numberOfTrajectories,
                        typename GPURandomWalkerBuilder<GPURandomWalker_t>::MoveFilterType moveFilterType,
                        uint32_t *intImageData, size_t width, size_t height,
                        typename GPURandomWalkerBuilder<GPURandomWalker_t>::BoundaryConditionsType
                        boundaryConditionsType, MoveFilter **moveFilter)
{
    if (!CUDA_IS_IT_FIRST_THREAD)
        return;

    using GPURandomWalkerBuilder_t = GPURandomWalkerBuilder<GPURandomWalker_t>;
    using MoveFilterType = typename GPURandomWalkerBuilder_t::MoveFilterType;
    using BoundaryConditionsType = typename GPURandomWalkerBuilder_t::BoundaryConditionsType;
    using DefaultMoveFilter = typename GPURandomWalkerBuilder_t::DefaultMoveFilter_t;
    using ImageMoveFilterWallBC = typename GPURandomWalkerBuilder_t::ImageMoveFilterWallBC_t;
    using ImageMoveFilterPeriodicBC = typename GPURandomWalkerBuilder_t::ImageMoveFilterPeriodicBC_t;

    if (moveFilterType == MoveFilterType::DEFAULT) {
        (*moveFilter) = new DefaultMoveFilter();
    } else if (moveFilterType == MoveFilterType::IMAGE) {
        if (boundaryConditionsType == BoundaryConditionsType::WALL)
            (*moveFilter) = new ImageMoveFilterWallBC(intImageData, width, height, seed, numberOfTrajectories);
        else if (boundaryConditionsType == BoundaryConditionsType::PERIODIC)
            (*moveFilter) = new ImageMoveFilterPeriodicBC(intImageData, width, height, seed, numberOfTrajectories);
        else
            (*moveFilter) = nullptr;
    } else {
        (*moveFilter) = nullptr;
    }
}

template<typename GPURandomWalker_t>
GPURandomWalkerBuilder<GPURandomWalker_t>::MoveGeneratorOnGPUFactory
    ::MoveGeneratorOnGPUFactory(const std::string &moveGeneratorString, float integrationStep)
            : integrationStep{integrationStep}
{
    Validate(integrationStep > 0.f);

    std::istringstream moveGeneratorStream(moveGeneratorString);
    std::string moveGeneratorName;
    moveGeneratorStream >> moveGeneratorName >> this->sigma;
    if (!moveGeneratorStream)
        throw std::runtime_error("Malformed MoveGenerator parameters");
    Validate(this->sigma > 0.f);

    if (moveGeneratorName == "GaussianMoveGenerator")
        this->moveGeneratorType = GAUSSIAN;
    else if (moveGeneratorName == "CauchyMoveGenerator")
        this->moveGeneratorType =  CAUCHY;
    else
        throw std::runtime_error("Unknown MoveGenerator: " + moveGeneratorName);
}

template<typename GPURandomWalker_t>
MoveGenerator *
GPURandomWalkerBuilder<GPURandomWalker_t>::MoveGeneratorOnGPUFactory
    ::create(unsigned long seed, std::size_t numberOfWalks)
{
    MoveGenerator **moveGeneratorPlaceholder{};
    cudaCheck( cudaMalloc(&moveGeneratorPlaceholder, sizeof(MoveGenerator**)) );
    create_move_generator<GPURandomWalker_t><<<1, 32>>>(seed, this->sigma, this->integrationStep, numberOfWalks,
                                                        this->moveGeneratorType, moveGeneratorPlaceholder);
    cudaCheck( cudaDeviceSynchronize() );

    MoveGenerator *moveGenerator;
    cudaCheck( cudaMemcpy(&moveGenerator, moveGeneratorPlaceholder, sizeof(MoveGenerator*),
                          cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(moveGeneratorPlaceholder) );

    return moveGenerator;
}

template<typename GPURandomWalker_t>
void
GPURandomWalkerBuilder<GPURandomWalker_t>::MoveFilterOnGPUFactory
    ::fetchImageData(std::istringstream &moveFilterStream, std::ostream &logger)
{
    std::string imageFilename;
    moveFilterStream >> imageFilename;
    if (!moveFilterStream)
        throw std::runtime_error("Malformed ImageMoveFilter parameters");

    this->fileIstreamProvider->setFileDescription("PPM image for MoveFilter");
    auto imageIstream = this->fileIstreamProvider->openFile(imageFilename);
    this->image = this->imageReader->read(*imageIstream);
    logger << "[GPURandomWalkerFactory] Loaded image " << imageFilename << " (" << this->image.getWidth();
    logger << "px x " << this->image.getHeight() << "px)" << std::endl;
}

template<typename GPURandomWalker_t>
void
GPURandomWalkerBuilder<GPURandomWalker_t>::MoveFilterOnGPUFactory
    ::fetchBoundaryConditions(std::istringstream &moveFilterStream)
{
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

template<typename GPURandomWalker_t>
GPURandomWalkerBuilder<GPURandomWalker_t>::MoveFilterOnGPUFactory
    ::MoveFilterOnGPUFactory(const std::string &moveFilterString, std::ostream &logger,
                             std::unique_ptr<FileIstreamProvider> fileIstreamProvider,
                             std::unique_ptr<ImageReader> imageReader)
            : fileIstreamProvider{std::move(fileIstreamProvider)}, imageReader{std::move(imageReader)}
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

template<typename GPURandomWalker_t>
MoveFilter *
GPURandomWalkerBuilder<GPURandomWalker_t>::MoveFilterOnGPUFactory
    ::create(unsigned long seed, std::size_t numberOfWalks)
{
    MoveFilter **moveFilterPlaceholder{};
    uint32_t *gpuIntImageData{};

    cudaCheck( cudaMalloc(&moveFilterPlaceholder, sizeof(MoveFilter**)) );

    auto intImageData = this->image.getIntData();
    if (this->moveFilterType == IMAGE) {
        cudaCheck( cudaMalloc(&gpuIntImageData, intImageData.size()*sizeof(uint32_t)));
        cudaCheck( cudaMemcpy(gpuIntImageData, intImageData.data(), intImageData.size()*sizeof(uint32_t),
                              cudaMemcpyHostToDevice) );
    }

    create_move_filter<GPURandomWalker_t><<<1, 32>>>(seed, numberOfWalks, this->moveFilterType, gpuIntImageData,
                                                     this->image.getWidth(), this->image.getHeight(),
                                                     this->boundaryConditionsType, moveFilterPlaceholder);
    cudaCheck( cudaDeviceSynchronize() );

    MoveFilter *moveFilter;
    cudaCheck( cudaMemcpy(&moveFilter, moveFilterPlaceholder, sizeof(MoveFilter*), cudaMemcpyDeviceToHost) );

    cudaCheck( cudaFree(moveFilterPlaceholder) );
    cudaCheck( cudaFree(gpuIntImageData) );

    return moveFilter;
}

template<typename GPURandomWalker_t>
GPURandomWalkerBuilder<GPURandomWalker_t>
    ::GPURandomWalkerBuilder(unsigned long seed, const RandomWalkerFactory::WalkerParameters &walkerParameters,
                             std::unique_ptr<FileIstreamProvider> fileIstreamProvider,
                             std::unique_ptr<ImageReader> imageReader, std::ostream &logger)
            : walkerParameters{walkerParameters}, numberOfWalksInSeries{walkerParameters.numberOfWalksInSeries},
              logger{logger},
              gpuMoveGeneratorFactory(walkerParameters.moveGeneratorParameters,
                                      walkerParameters.walkParameters.integrationStep),
              gpuMoveFilterFactory(walkerParameters.moveFilterParameters, logger, std::move(fileIstreamProvider),
                                  std::move(imageReader)), seedGenerator(seed)
{ }

template<typename GPURandomWalker_t>
std::unique_ptr<RandomWalker> GPURandomWalkerBuilder<GPURandomWalker_t>::build() {
    MoveGenerator *moveGenerator = this->gpuMoveGeneratorFactory.create(this->seedGenerator(),
                                                                        this->numberOfWalksInSeries);
    MoveFilter *moveFilter = this->gpuMoveFilterFactory.create(this->seedGenerator(), this->numberOfWalksInSeries);

    return std::unique_ptr<RandomWalker>(
        new GPURandomWalker_t(this->numberOfWalksInSeries, walkerParameters.walkParameters,
                              gpuMoveFilterFactory.numberOfSetupThreads, moveGenerator, moveFilter, logger)
    );
}
