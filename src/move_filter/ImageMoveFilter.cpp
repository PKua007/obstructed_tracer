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

ImageMoveFilter::ImageMoveFilter(Image image, unsigned int seed) :
        width{image.getWidth()}, height{image.getHeight()} {
    this->randomGenerator.seed(seed);
    this->validPointsMap.reserve(image.getNumberOfPixels());

    // Image y axis starts from left upper corner downwards, so image is scanned from the bottom left, because
    // validPointsMap is in "normal" coordinate system, with (0, 0) in left bottom corner
    for (std::size_t y = 0; y < this->height; y++) {
        for (std::size_t x = 0; x < this->width; x++) {
            if (image(x, this->height - y - 1) == WHITE) {
                this->validPointsIndices.push_back(this->validPointsMap.size());
                this->validPointsMap.push_back(true);
            } else {
                this->validPointsMap.push_back(false);
            }
        }
    }

    if (this->validPointsIndices.empty())
        throw std::runtime_error("No valid points found in a given image");
    std::cout << "[ImageMoveFilter::ImageMoveFilter] Found " << this->validPointsIndices.size();
    std::cout << " valid starting pixels out of " << this->validPointsMap.size() << " total" << std::endl;
}
bool ImageMoveFilter::isPointValid(ImagePoint point) const {
    if (point.x < 0 || point.x >= this->width || point.y < 0 || point.y >= this->height)
        return false;

    return this->validPointsMap[point.x + point.y * this->width];
}

bool ImageMoveFilter::isLineValid(ImagePoint from, ImagePoint to) const {
    ImageMove imageMove = to - from;
    if (std::abs(imageMove.x) > std::abs(imageMove.y)) {
        float a = float(imageMove.y) / float(imageMove.x);
        for (int x = from.x; x != to.x; x += sgn(imageMove.x)) {
            int y = static_cast<int>(from.y + a * (x - from.x));
            if (!this->isPointValid( { x, y }))
                return false;
        }
    } else {
        float a = float(imageMove.x) / float(imageMove.y);
        for (int y = from.y; y != to.y; y += sgn(imageMove.y)) {
            int x = static_cast<int>(from.x + a * (y - from.y));
            if (!this->isPointValid( { x, y }))
                return false;
        }
    }
    return true;
}

ImageMoveFilter::ImagePoint ImageMoveFilter::indexToPoint(std::size_t index) const {
    Expects(index < this->validPointsMap.size());
    return {static_cast<int>(index % this->width), static_cast<int>(index / this->width)};
}


bool ImageMoveFilter::isMoveValid(Point from, Move move) const {
    Point to = from + move;
    ImagePoint imageFrom(from);
    ImagePoint imageTo(to);

    if (imageFrom == imageTo)
        return true;

    if (!isPointValid(imageTo))
        return false;

    return isLineValid(imageFrom, imageTo);
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
