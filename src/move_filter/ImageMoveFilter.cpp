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

bool ImageMoveFilter::isMoveValid(Point tracer, Move move) const {
    Point finalTracer = tracer + move;

    if (finalTracer.x < 0 || finalTracer.x >= this->width || finalTracer.y < 0 || finalTracer.y >= this->height)
        return false;

    ImagePoint imageFinalTracer = {static_cast<std::size_t>(finalTracer.x), static_cast<std::size_t>(finalTracer.y)};
    Assert(imageFinalTracer.x < this->width && imageFinalTracer.y < this->height);

    if (!isPointValid(imageFinalTracer))
        return false;

    int x1 = tracer.x, x2 = finalTracer.x;
    int y1 = tracer.y, y2 = finalTracer.y;
    if (x1==x2 && y1==y2) return true;
    if (std::abs(x2-x1)>std::abs(y2-y1)){
        float a = float(y2-y1)/(x2-x1);
        for(int x=(int)x1; x!=(int)x2; x += sgn(x2-x1)){
            int y = (int)(y1 + a*(x-x1));
            if (!this->isPointValid({x, y}))
                return false;
        }
    }else{
        float a = float(x2-x1)/(y2-y1);
        for(int y=(int)y1; y!=(int)y2; y += sgn(y2-y1)){
            int x = (int)(x1 + a*(y-y1));
            if (!this->isPointValid({x, y}))
                return false;
        }

    }
    return true;
}

bool ImageMoveFilter::isPointValid(ImagePoint point) const {
    Expects(point.x < this->width);
    Expects(point.y < this->height);

    return this->validPointsMap[point.x + point.y * this->width];
}

ImageMoveFilter::ImagePoint ImageMoveFilter::indexToPoint(std::size_t index) const {
    Expects(index < this->validPointsMap.size());
    return {index % this->width, index / this->width};
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
