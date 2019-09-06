/*
 * ImageMoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef IMAGEMOVEFILTER_H_
#define IMAGEMOVEFILTER_H_

#include <random>
#include <curand_kernel.h>

#include "simulation/MoveFilter.h"
#include "ImageBoundaryConditions.h"
#include "ImagePoint.h"

class ImageMoveFilter: public MoveFilter {
private:

#if CUDA_DEVICE_COMPILATION
    curandState *states;
    size_t numberOfStates;
#else // CUDA_HOST_COMPILATION
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<float> uniformDistribution{0.f, 1.f};

    std::vector<size_t> validTracerIndicesCache{};
#endif

    float tracerRadius{};
    size_t width{};
    size_t height{};
    ImageBoundaryConditions *imageBC{};
    bool* validPointsMap{};
    bool* validTracersMap{};
    size_t validPointsMapSize{};

    CUDA_HOSTDEV void initializeGenerators(unsigned long seed, size_t numberOfTrajectories);
    CUDA_HOSTDEV bool isPointValid(ImagePoint point, float pointRadius) const;
    CUDA_HOSTDEV bool checkValidPointsMap(ImagePoint point) const;
    CUDA_HOSTDEV bool checkValidTracersMap(ImagePoint point) const;
    CUDA_HOSTDEV bool isLineValid(ImagePoint from, ImagePoint to) const;
    CUDA_HOSTDEV ImagePoint indexToImagePoint(std::size_t index) const;
    CUDA_HOSTDEV size_t pointToIndex(ImagePoint point) const;
    CUDA_HOSTDEV float randomUniformNumber();
    CUDA_HOSTDEV ImagePoint randomTracerImagePosition();

public:
    CUDA_HOSTDEV ImageMoveFilter(unsigned int *intImageData, size_t width, size_t height,
                                 ImageBoundaryConditions *imageBC, unsigned long seed, size_t numberOfTrajectories);

    CUDA_HOSTDEV ImageMoveFilter(const ImageMoveFilter &other) = delete;
    CUDA_HOSTDEV ImageMoveFilter operator=(ImageMoveFilter other) = delete;
    CUDA_HOSTDEV ~ImageMoveFilter();

    CUDA_HOSTDEV bool isMoveValid(Tracer tracer, Move move) const override;
    CUDA_HOSTDEV Tracer randomValidTracer() override;
    CUDA_HOSTDEV void setupForTracerRadius(float radius) override;

    CUDA_HOSTDEV size_t getNumberOfAllPoints() const;

#if CUDA_HOST_COMPILATION
    std::size_t getNumberOfValidTracers();
#endif
};

#endif /* IMAGEMOVEFILTER_H_ */
