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
#include "utils/CudaQualifiers.h"

class ImageMoveFilter: public MoveFilter {
private:

#ifdef __CUDA_ARCH__
    curandState *states;
    size_t numberOfStates;
#else
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<float> uniformDistribution{0.f, 1.f};
#endif

    size_t width{};
    size_t height{};
    ImageBoundaryConditions *imageBC{};
    bool* validPointsMap{};
    size_t validPointsMapSize{};

    float radiusForTracerCache = -1.0;
    size_t *validTracerIndicesCache{};
    size_t validTracerIndicesCacheSize{};

    CUDA_HOSTDEV bool isPointValid(ImagePoint point, float pointRadius) const;
    CUDA_HOSTDEV bool checkValidPointsMap(ImagePoint point) const;
    CUDA_HOSTDEV bool isLineValid(ImagePoint from, ImagePoint to, float pointRadius) const;
    CUDA_HOSTDEV ImagePoint indexToPoint(std::size_t index) const;
    CUDA_HOSTDEV size_t pointToIndex(ImagePoint point) const;
    CUDA_HOSTDEV float randomUniformNumber();

public:
    CUDA_HOSTDEV ImageMoveFilter(unsigned int *intImageData, size_t width, size_t height,
                                 ImageBoundaryConditions *imageBC, unsigned long seed, size_t numberOfTrajectories);

    CUDA_HOSTDEV ImageMoveFilter(const ImageMoveFilter &other) = delete;
    CUDA_HOSTDEV ImageMoveFilter operator=(ImageMoveFilter other) = delete;
    CUDA_HOSTDEV ~ImageMoveFilter();

    CUDA_HOSTDEV bool isMoveValid(Tracer tracer, Move move) const override;
    CUDA_HOSTDEV Tracer randomValidTracer(float radius) override;

    CUDA_HOSTDEV size_t getNumberOfAllPoints() const;
    CUDA_HOSTDEV size_t getNumberOfValidTracers(float radius);
    CUDA_HOSTDEV void rebuildValidTracersCache(float radius);
};

#endif /* IMAGEMOVEFILTER_H_ */
