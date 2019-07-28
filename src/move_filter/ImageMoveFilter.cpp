/*
 * ImageMoveFilter.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include "ImageMoveFilter.h"

ImageMoveFilter::ImageMoveFilter(Image image, unsigned int seed) :
        image{image} {
    this->randomGenerator.seed(seed);
}

bool ImageMoveFilter::isMoveValid(Point tracer, Move move) const {
    return true;
}

Point ImageMoveFilter::randomValidPoint() const {
    return Point{};
}
