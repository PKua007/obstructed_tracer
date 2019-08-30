/*
 * GPUSimulationFactory.cpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include <sstream>
#include <fstream>
#include <vector>

#include "GPUSimulationFactory.h"
#include "utils/CudaCheck.h"
#include "move_generator/GPUGaussianMoveGenerator.h"
#include "move_filter/DefaultMoveFilter.h"
#include "move_filter/image_move_filter/ImageMoveFilter.h"
#include "move_filter/image_move_filter/WallBoundaryConditions.h"
#include "move_filter/image_move_filter/PeriodicBoundaryConditions.h"
#include "image/PPMImageReader.h"
#include "utils/Assertions.h"

namespace {
    enum MoveGeneratorType {
        GAUSSIAN,
        CAUCHY
    };

    enum MoveFilterType {
        DEFAULT,
        IMAGE
    };

    enum BoundaryConditionsType {
        WALL,
        PERIODIC
    };

    __global__
    void create_move_filter(unsigned long seed, size_t numberOfTrajectories, MoveFilterType moveFilterType,
                            uint32_t *intImageData, size_t width, size_t height,
                            BoundaryConditionsType boundaryConditionsType, MoveFilter **moveFilter,
                            ImageBoundaryConditions **boundaryConditions)
    {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i != 0)
            return;

        if (moveFilterType == IMAGE) {
            if (boundaryConditionsType == WALL)
                (*boundaryConditions) = new WallBoundaryConditions();
            else if (boundaryConditionsType == PERIODIC)
                (*boundaryConditions) = new PeriodicBoundaryConditions();
            else
                (*boundaryConditions) = nullptr;
        } else {
            (*boundaryConditions) = nullptr;
        }

        if (moveFilterType == DEFAULT)
            (*moveFilter) = new DefaultMoveFilter();
        else if (moveFilterType == IMAGE)
            (*moveFilter) = new ImageMoveFilter(intImageData, width, height, *boundaryConditions, seed,
                                                numberOfTrajectories);
        else
            (*moveFilter) = nullptr;
    }



    __global__
    void create_move_generator(unsigned long seed, float sigma, size_t numberOfTrajectories,
                               MoveGeneratorType moveGeneratorType, MoveGenerator **moveGenerator)
    {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i != 0)
            return;

        if (moveGeneratorType == GAUSSIAN)
            (*moveGenerator) = new GPUGaussianMoveGenerator(sigma, seed, numberOfTrajectories);
        else if (moveGeneratorType == CAUCHY)
            (*moveGenerator) = nullptr;
        else
            (*moveGenerator) = nullptr;
    }

    class MoveGeneratorOnGPU {
    private:
        MoveGeneratorType moveGeneratorType{};
        float sigma{};

    public:
        MoveGenerator *moveGenerator{};

        MoveGeneratorOnGPU(const Parameters &parameters) {
            std::istringstream moveGeneratorStream(parameters.moveGenerator);
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

        void allocateOnGPU(unsigned long seed, std::size_t numberOfWalks) {
            MoveGenerator **moveGeneratorPlaceholder{};
            cudaCheck( cudaMalloc(&moveGeneratorPlaceholder, sizeof(MoveGenerator**)) );
            create_move_generator<<<1, 32>>>(seed, this->sigma, numberOfWalks, this->moveGeneratorType,
                                             moveGeneratorPlaceholder);
            cudaCheck( cudaDeviceSynchronize() );
            cudaCheck( cudaMemcpy(&(this->moveGenerator), moveGeneratorPlaceholder, sizeof(MoveGenerator*),
                                  cudaMemcpyDeviceToHost) );
            cudaCheck( cudaFree(moveGeneratorPlaceholder) );
        }
    };

    class MoveFilterOnGPU {
    private:
        MoveFilterType moveFilterType{};
        BoundaryConditionsType boundaryConditionsType{};
        Image image{};

        void fetchImageData(std::istringstream &moveFilterStream, std::ostream &logger) {
            std::string imageFilename;
            moveFilterStream >> imageFilename;
            if (!moveFilterStream)
                throw std::runtime_error("Malformed ImageMoveFilter parameters");

            std::ifstream imageFile(imageFilename);
            if (!imageFile)
                throw std::runtime_error("Cannot open " + imageFilename + " to load image");

            PPMImageReader imageReader;
            this->image = imageReader.read(imageFile);
            logger << "[GPUSimulationFactory] Loaded image " << imageFilename << " (" << this->image.getWidth();
            logger << "px x " << this->image.getHeight() << "px)" << std::endl;
        }

        void fetchBoundaryConditions(std::istringstream &moveFilterStream) {
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

    public:
        MoveFilter *moveFilter{};
        ImageBoundaryConditions *boundaryConditions{};

        MoveFilterOnGPU(const Parameters &parameters, std::ostream &logger) {
            std::istringstream moveFilterStream(parameters.moveFilter);
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
            }
        }

        void allocateOnGPU(unsigned long seed, std::size_t numberOfWalks) {
            MoveFilter **moveFilterPlaceholder{};
            ImageBoundaryConditions **boundaryConditionsPlaceholder{};
            uint32_t *gpuIntImageData{};

            cudaCheck( cudaMalloc(&moveFilterPlaceholder, sizeof(MoveFilter**)) );
            cudaCheck( cudaMalloc(&boundaryConditionsPlaceholder, sizeof(ImageBoundaryConditions**)) );

            auto intImageData = this->image.getIntData();
            if (this->moveFilterType == IMAGE) {
                cudaCheck( cudaMalloc(&gpuIntImageData, intImageData.size()*sizeof(uint32_t)));
                cudaCheck( cudaMemcpy(gpuIntImageData, intImageData.data(), intImageData.size()*sizeof(uint32_t),
                                      cudaMemcpyHostToDevice) );
            }

            create_move_filter<<<1, 32>>>(seed, numberOfWalks, this->moveFilterType, gpuIntImageData,
                                          this->image.getWidth(), this->image.getHeight(), this->boundaryConditionsType,
                                          moveFilterPlaceholder, boundaryConditionsPlaceholder);
            cudaCheck( cudaDeviceSynchronize() );

            cudaCheck( cudaMemcpy(&(this->moveFilter), moveFilterPlaceholder, sizeof(MoveFilter*),
                                  cudaMemcpyDeviceToHost) );
            cudaCheck( cudaMemcpy(&(this->boundaryConditions), boundaryConditionsPlaceholder,
                                  sizeof(ImageBoundaryConditions*), cudaMemcpyDeviceToHost) );

            cudaCheck( cudaFree(moveFilterPlaceholder) );
            cudaCheck( cudaFree(boundaryConditionsPlaceholder) );
            cudaCheck( cudaFree(gpuIntImageData) );
        }
    };
}


__global__
void delete_objects(MoveGenerator *moveGenerator, MoveFilter *moveFilter, ImageBoundaryConditions *boundaryConditions)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i != 0)
        return;

    delete moveGenerator;
    delete moveFilter;
    delete boundaryConditions;
}

void GPUSimulationFactory::initializeSeedGenerator(const Parameters &parameters, std::ostream &logger) {
    if (parameters.seed == "random") {
        unsigned long randomSeed = std::random_device()();
        this->seedGenerator.seed(randomSeed);
        logger << "[GPUSimulationFactory] Using random seed: " << randomSeed << std::endl;
    } else {
        this->seedGenerator.seed(std::stoul(parameters.seed));
    }
}

GPUSimulationFactory::GPUSimulationFactory(const Parameters& parameters, std::ostream& logger) {
    this->initializeSeedGenerator(parameters, logger);

    MoveGeneratorOnGPU gpuMoveGenerator(parameters);
    MoveFilterOnGPU gpuMoveFilter(parameters, logger);

    gpuMoveGenerator.allocateOnGPU(this->seedGenerator(), parameters.numberOfWalks);
    gpuMoveFilter.allocateOnGPU(this->seedGenerator(), parameters.numberOfWalks);

    this->moveGenerator = gpuMoveGenerator.moveGenerator;
    this->moveFilter = gpuMoveFilter.moveFilter;
    this->imageBoundaryConditions = gpuMoveFilter.boundaryConditions;

    Move drift = {parameters.driftX, parameters.driftY};
    this->randomWalker.reset(new GPURandomWalker(parameters.numberOfWalks, parameters.numberOfSteps,
                                                 parameters.tracerRadius, drift, this->moveGenerator,
                                                 this->moveFilter));
}

GPUSimulationFactory::~GPUSimulationFactory() {
    delete_objects<<<1, 32>>>(this->moveGenerator, this->moveFilter, this->imageBoundaryConditions);
    cudaCheck( cudaDeviceSynchronize() );
}

RandomWalker& GPUSimulationFactory::getRandomWalker() {
    return *this->randomWalker;
}
