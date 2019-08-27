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

        ImageMove() = default;
        ImageMove(int x, int y) : x{x}, y{y} { };
    };

    ImageMove operator-(ImagePoint p1, ImagePoint p2) {
        return {p1.x - p2.x, p1.y - p2.y};
    }


}

ImageMoveFilter::ImageMoveFilter(Image image, ImageBoundaryConditions *imageBC, unsigned int seed) :
        width{image.getWidth()}, height{image.getHeight()}, imageBC{imageBC} {
    this->randomGenerator.seed(seed);
    this->validPointsMapSize = image.getNumberOfPixels();
    this->validPointsMap = new bool[this->validPointsMapSize];
    this->validTracerIndicesCache = new size_t[this->validPointsMapSize];
    this->imageBC->installOnImage(image);

    // Image y axis starts from left upper corner downwards, so image is scanned from the bottom left, because
    // validPointsMap is in "normal" coordinate system, with (0, 0) in left bottom corner
    size_t i = 0;
    for (size_t y = 0; y < this->height; y++) {
        for (size_t x = 0; x < this->width; x++) {
            if (image(x, this->height - y - 1) == WHITE)
                this->validPointsMap[i] = true;
            else
                this->validPointsMap[i] = false;

            i++;
        }
    }
}

ImageMoveFilter::~ImageMoveFilter() {
    delete [] this->validPointsMap;
    delete [] this->validTracerIndicesCache;
}

void ImageMoveFilter::rebuildValidTracersCache(float radius) {
    Expects(radius >= 0.f);

    if (this->radiusForTracerCache == radius)
        return;

    this->radiusForTracerCache = radius;
    this->validTracerIndicesCacheSize = 0;
    for (size_t i = 0; i < this->validPointsMapSize; i++) {
        if (this->isPointValid(this->indexToPoint(i), radius)) {
            this->validTracerIndicesCache[this->validTracerIndicesCacheSize] = i;
            this->validTracerIndicesCacheSize++;
        }
    }
}

bool ImageMoveFilter::checkValidPointsMap(ImagePoint point) const {
    point = this->imageBC->applyOnImagePoint(point);
    return this->validPointsMap[point.x + point.y * this->width];
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

Tracer ImageMoveFilter::randomValidTracer(float radius) {
    Expects(radius >= 0.f);

    this->rebuildValidTracersCache(radius);
    if (this->validTracerIndicesCacheSize == 0)
        throw std::runtime_error("No valid points found in a given image");

    float floatCacheIndex = this->uniformDistribution(this->randomGenerator) * this->validTracerIndicesCacheSize;
    size_t cacheIndex = static_cast<size_t>(floatCacheIndex);
    Assert(cacheIndex < this->validTracerIndicesCacheSize);
    size_t tracerIndex = this->validTracerIndicesCache[cacheIndex];
    ImagePoint imagePosition = this->indexToPoint(tracerIndex);

    float pixelOffsetX = this->uniformDistribution(this->randomGenerator);
    float pixelOffsetY = this->uniformDistribution(this->randomGenerator);

    Point tracerPosition = {imagePosition.x + pixelOffsetX, imagePosition.y + pixelOffsetY};
    return Tracer(tracerPosition, radius);
}

size_t ImageMoveFilter::getNumberOfAllPoints() const {
    return this->validPointsMapSize;
}

size_t ImageMoveFilter::getNumberOfValidTracers(float radius) {
    this->rebuildValidTracersCache(radius);
    return this->validTracerIndicesCacheSize;
}
