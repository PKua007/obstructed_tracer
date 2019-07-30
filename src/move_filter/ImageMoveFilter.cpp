/*
 * ImageMoveFilter.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include <iostream>

#include "ImageMoveFilter.h"
#include "../utils/Assertions.h"
#include "../utils/Utils.h"


bool ImageMoveFilter::ImagePoint::operator==(ImageMoveFilter::ImagePoint second) const {
    return this->x == second.x && this->y == second.y;
}

bool ImageMoveFilter::ImagePoint::operator!=(ImageMoveFilter::ImagePoint second) const {
    return !(*this == second);
}


ImageMoveFilter::ImageMove ImageMoveFilter::ImagePoint::operator-(ImagePoint second) const {
    return {this->x - second.x, this->y - second.y};
}

ImageMoveFilter::ImageMoveFilter(Image image, float tracerRadius, unsigned int seed) :
        width{image.getWidth()}, height{image.getHeight()} {
    this->randomGenerator.seed(seed);
    this->validPointsMap.reserve(image.getNumberOfPixels());

    // Image y axis starts from left upper corner downwards, so image is scanned from the bottom left, because
    // validPointsMap is in "normal" coordinate system, with (0, 0) in left bottom corner
    for (std::size_t y = 0; y < this->height; y++) {
        for (std::size_t x = 0; x < this->width; x++) {
            if (image(x, this->height - y - 1) == WHITE)
                this->validPointsMap.push_back(true);
            else
                this->validPointsMap.push_back(false);
        }
    }

    for (std::size_t i = 0; i < this->validPointsMap.size(); i++)
        if (this->isPointValid(this->indexToPoint(i), tracerRadius))
            this->validPointsIndices.push_back(i);

    if (this->validPointsIndices.empty())
        throw std::runtime_error("No valid points found in a given image");
    std::cout << "[ImageMoveFilter::ImageMoveFilter] Found " << this->validPointsIndices.size();
    std::cout << " valid starting pixels out of " << this->validPointsMap.size() << " total" << std::endl;
}

bool ImageMoveFilter::isPointValid(ImagePoint point) const {
    return this->validPointsMap[point.x + point.y * this->width];
}

bool ImageMoveFilter::isPointValid(ImagePoint point, float pointRadius) const {
    Expects(pointRadius >= 0.f);

    int intPointRadius = static_cast<int>(pointRadius);
    if (point.x - intPointRadius < 0 || point.x + intPointRadius >= this->width)
        return false;
    if (point.y - intPointRadius < 0 || point.y + intPointRadius >= this->height)
        return false;

    if (pointRadius == 0.f)
        return this->isPointValid(point);

    for (int x = -intPointRadius; x <= intPointRadius; x++) {
        for (int y = -intPointRadius; y <= intPointRadius; y++) {
            if (x*x + y*y > pointRadius*pointRadius)
                continue;

            if (!this->isPointValid({point.x + x, point.y + y}))
                return false;
        }
    }
    return true;
}

bool ImageMoveFilter::isLineValid(ImagePoint from, ImagePoint to, float pointRadius) const {
    ImageMove imageMove = to - from;
    if (std::abs(imageMove.x) > std::abs(imageMove.y)) {
        float a = float(imageMove.y) / float(imageMove.x);
        for (int x = from.x; x != to.x; x += sgn(imageMove.x)) {
            int y = static_cast<int>(from.y + a * (x - from.x));
            if (!this->isPointValid({ x, y }, pointRadius))
                return false;
        }
    } else {
        float a = float(imageMove.x) / float(imageMove.y);
        for (int y = from.y; y != to.y; y += sgn(imageMove.y)) {
            int x = static_cast<int>(from.x + a * (y - from.y));
            if (!this->isPointValid({ x, y }, pointRadius))
                return false;
        }
    }
    return true;
}

ImageMoveFilter::ImagePoint ImageMoveFilter::indexToPoint(std::size_t index) const {
    Expects(index < this->validPointsMap.size());
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

Point ImageMoveFilter::randomValidPoint() {
    float randomPixelFloatIndex = this->uniformDistribution(this->randomGenerator) * this->validPointsIndices.size();
    std::size_t randomPixelIndex = static_cast<std::size_t>(randomPixelFloatIndex);
    Assert(randomPixelIndex < this->validPointsIndices.size());
    std::size_t randomValidPointIndex = this->validPointsIndices[randomPixelIndex];
    ImagePoint randomValidPoint = this->indexToPoint(randomValidPointIndex);

    float randomPixelOffsetX = this->uniformDistribution(this->randomGenerator);
    float randomPixelOffsetY = this->uniformDistribution(this->randomGenerator);

    return {randomValidPoint.x + randomPixelOffsetX, randomValidPoint.y + randomPixelOffsetY};
}
