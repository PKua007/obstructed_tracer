/*
 * ImageMoveFilter.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include <iostream>

#include "ImageMoveFilter.h"
#include "utils/Assertions.h"
#include "utils/Utils.h"

namespace {
    struct ImageMove {
        int x{};
        int y{};

        CUDA_HOSTDEV ImageMove() = default;
        CUDA_HOSTDEV ImageMove(int x, int y) : x{x}, y{y} { };
    };

    CUDA_HOSTDEV ImageMove operator-(ImagePoint p1, ImagePoint p2) {
        return {p1.x - p2.x, p1.y - p2.y};
    }

    template <typename T> CUDA_HOSTDEV int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }
}


#pragma nv_exec_check_disable

ImageMoveFilter::ImageMoveFilter(unsigned int *intImageData, size_t width, size_t height,
                                 ImageBoundaryConditions *imageBC, unsigned long seed, size_t numberOfTrajectories) :
        width{width}, height{height}, imageBC{imageBC} {
    #if CUDA_DEVICE_COMPILATION
        this->states = new curandState[numberOfTrajectories];
        for (size_t i = 0; i < numberOfTrajectories; i++)
            curand_init(seed, i, 0, &(this->states[i]));
    #else // CUDA_HOST_COMPILATION
        this->randomGenerator.seed(seed);
    #endif

    this->validPointsMapSize = this->width * this->height;
    this->validPointsMap = new bool[this->validPointsMapSize];
    this->imageBC->setupDimensions(this->width, this->height);

    // Image y axis starts from left upper corner downwards, so image is scanned from the bottom left, because
    // validPointsMap is in "normal" coordinate system, with (0, 0) in left bottom corner
    size_t i = 0;
    for (size_t y = 0; y < this->height; y++) {
        for (size_t x = 0; x < this->width; x++) {
            ImagePoint imagePoint = {static_cast<int>(x), static_cast<int>(this->height - y - 1)};
            if (intImageData[this->pointToIndex(imagePoint)] == 0xffffffff)
                this->validPointsMap[i] = true;
            else
                this->validPointsMap[i] = false;

            i++;
        }
    }
}

ImageMoveFilter::~ImageMoveFilter() {
    delete [] this->validPointsMap;
    #if CUDA_DEVICE_COMPILATION
        delete [] this->states;
    #endif
}

bool ImageMoveFilter::checkValidPointsMap(ImagePoint point) const {
    point = this->imageBC->applyOnImagePoint(point);
    return this->validPointsMap[this->pointToIndex(point)];
}

bool ImageMoveFilter::isPointValid(ImagePoint point, float pointRadius) const {
    Expects(pointRadius >= 0.f);

    int intPointRadius = static_cast<int>(pointRadius);
    if (!this->imageBC->isImagePointInBounds(point, intPointRadius))
        return false;

    if (pointRadius == 0.f)
        return this->checkValidPointsMap(point);

    for (int x = -intPointRadius; x <= intPointRadius; x++) {
        for (int y = -intPointRadius; y <= intPointRadius; y++) {
            if (x*x + y*y > pointRadius*pointRadius)
                continue;

            if (!this->checkValidPointsMap({point.x + x, point.y + y}))
                return false;
        }
    }
    return true;
}

bool ImageMoveFilter::isLineValid(ImagePoint from, ImagePoint to, float pointRadius) const {
    ImageMove imageMove = to - from;
    if (abs(imageMove.x) > abs(imageMove.y)) {
        float a = float(imageMove.y) / float(imageMove.x);
        for (int x = from.x; x != to.x; x += sgn(imageMove.x)) {
            int y = static_cast<int>(round(from.y + a * (x - from.x)));
            if (!this->isPointValid({ x, y }, pointRadius))
                return false;
        }
    } else {
        float a = float(imageMove.x) / float(imageMove.y);
        for (int y = from.y; y != to.y; y += sgn(imageMove.y)) {
            int x = static_cast<int>(round(from.x + a * (y - from.y)));
            if (!this->isPointValid({ x, y }, pointRadius))
                return false;
        }
    }
    return true;
}

ImagePoint ImageMoveFilter::indexToPoint(size_t index) const {
    Expects(index < this->validPointsMapSize);
    return {static_cast<int>(index % this->width), static_cast<int>(index / this->width)};
}

size_t ImageMoveFilter::pointToIndex(ImagePoint point) const {
    return point.x + this->width * point.y;
}

#pragma nv_exec_check_disable
float ImageMoveFilter::randomUniformNumber() {
    #if CUDA_DEVICE_COMPILATION
        // 1 minus curand_normal, because it samples from (0, 1], and we want [0, 1)
        return 1.f - curand_uniform(&(this->states[CUDA_THREAD_IDX]));
    #else // CUDA_HOST_COMPILATION
        return this->uniformDistribution(this->randomGenerator);
    #endif
}

bool ImageMoveFilter::isMoveValid(Tracer tracer, Move move) const {
    Point from = tracer.getPosition();
    Point to = from + move;
    ImagePoint imageFrom(from);
    ImagePoint imageTo(to);

    if (imageFrom == imageTo)
        return true;

    if (!isPointValid(imageTo, tracer.getRadius()))
        return false;

    return isLineValid(imageFrom, imageTo, tracer.getRadius());
}

#pragma nv_exec_check_disable
Tracer ImageMoveFilter::randomValidTracer(float radius) {
    Expects(radius >= 0.f);

    ImagePoint imagePosition;

    #if CUDA_DEVICE_COMPILATION
        do {
            float floatMapIndex = this->randomUniformNumber() * this->validPointsMapSize;
            size_t mapIndex = static_cast<size_t>(floatMapIndex);
            imagePosition = this->indexToPoint(mapIndex);
        } while(!this->isPointValid(imagePosition, radius));
    #else // CUDA_HOST_COMPILATION
        this->rebuildValidTracersCache(radius);
        if (this->validTracerIndicesCache.empty())
            throw std::runtime_error("No valid points found in a given image");

        float floatCacheIndex = this->randomUniformNumber() * this->validTracerIndicesCache.size();
        size_t cacheIndex = static_cast<size_t>(floatCacheIndex);
        Assert(cacheIndex < this->validTracerIndicesCacheSize);
        size_t tracerIndex = this->validTracerIndicesCache[cacheIndex];
        imagePosition = this->indexToPoint(tracerIndex);
    #endif

    float pixelOffsetX = this->randomUniformNumber();
    float pixelOffsetY = this->randomUniformNumber();

    Point tracerPosition = {imagePosition.x + pixelOffsetX, imagePosition.y + pixelOffsetY};
    return Tracer(tracerPosition, radius);
}

size_t ImageMoveFilter::getNumberOfAllPoints() const {
    return this->validPointsMapSize;
}


#if CUDA_HOST_COMPILATION

void ImageMoveFilter::rebuildValidTracersCache(float radius) {
    Expects(radius >= 0.f);

    if (this->radiusForTracerCache == radius)
        return;

    this->radiusForTracerCache = radius;
    this->validTracerIndicesCache.clear();
    for (size_t i = 0; i < this->validPointsMapSize; i++)
        if (this->isPointValid(this->indexToPoint(i), radius))
            this->validTracerIndicesCache.push_back(i);
}

size_t ImageMoveFilter::getNumberOfValidTracers(float radius) {
    this->rebuildValidTracersCache(radius);
    return this->validTracerIndicesCache.size();
}

#endif /* HOST_COMPILATION */
