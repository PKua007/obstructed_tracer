/*
 * ImageMoveFilter.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include <iostream>

#include "ImageMoveFilter.h"
#include "../utils/Assertions.h"

ImageMoveFilter::ImageMoveFilter(Image image, unsigned int seed) :
        image{image} {
    this->randomGenerator.seed(seed);

    for (std::size_t i = 0; i < this->image.getWidth(); i++)
        for (std::size_t j = 0; j < this->image.getHeight(); j++)
            if (this->image(i, j) == WHITE)
                this->validPoints.push_back(ImagePoint{i, j});

    if (this->validPoints.empty())
        throw std::runtime_error("No valid points found in a given image");
    std::cout << "[ImageMoveFilter::ImageMoveFilter] Found " << validPoints.size() << " valid starting pixels out of ";
    std::cout << this->image.getNumberOfPixels() << " total" << std::endl;
}

bool ImageMoveFilter::isMoveValid(Point tracer, Move move) const {
    Point finalTracer = tracer + move;

    float width = this->image.getWidth();
    float height = this->image.getHeight();
    if (finalTracer.x < 0 || finalTracer.x >= width || finalTracer.y < 0 || finalTracer.y >= height)
        return false;

    ImagePoint imageFinalTracer = {static_cast<std::size_t>(finalTracer.x), static_cast<std::size_t>(finalTracer.y)};
    Assert(imageFinalTracer.x < width && imageFinalTracer.y < height);

    // Image y axis starts from left upper corner downwards. We use here lower left upwards instead
    if (this->image(imageFinalTracer.x, height - 1 - imageFinalTracer.y) != WHITE)
        return false;

    return true;
}

Point ImageMoveFilter::randomValidPoint() {
    float randomPixelFloatIndex = this->uniformDistribution(this->randomGenerator) * this->validPoints.size();
    std::size_t randomPixelIndex = static_cast<std::size_t>(randomPixelFloatIndex);
    Assert(randomPixelIndex < this->validPoints.size());
    ImagePoint randomValidPoint = this->validPoints[randomPixelIndex];

    // Image y axis starts from left upper corner downwards. We use here lower left upwards instead
    randomValidPoint.y = this->image.getHeight() - 1 - randomValidPoint.y;

    float randomPixelOffsetX = this->uniformDistribution(this->randomGenerator);
    float randomPixelOffsetY = this->uniformDistribution(this->randomGenerator);

    return {randomValidPoint.x + randomPixelOffsetX, randomValidPoint.y + randomPixelOffsetY};
}
