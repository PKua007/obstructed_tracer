/*
 * PeriodicBoundaryConditions.cpp
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#include "PeriodicBoundaryConditions.h"

void PeriodicBoundaryConditions::setupDimensions(size_t width, size_t height) {
    this->width = width;
    this->height = height;
}

bool PeriodicBoundaryConditions::isImagePointInBounds(ImagePoint imagePoint, int radius) const {
    return true;
}

ImagePoint PeriodicBoundaryConditions::applyOnImagePoint(ImagePoint imagePoint) const {
    return {mod(imagePoint.x, this->width), mod(imagePoint.y, this->height)};
}
