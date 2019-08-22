/*
 * PeriodicBoundaryConditions.cpp
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#include "PeriodicBoundaryConditions.h"

void PeriodicBoundaryConditions::installOnImage(const Image& image) {
    this->width = image.getWidth();
    this->height = image.getHeight();
}

bool PeriodicBoundaryConditions::isImagePointInBounds(ImagePoint imagePoint, int radius) const {
    return true;
}

ImagePoint PeriodicBoundaryConditions::applyOnImagePoint(ImagePoint imagePoint) const {
    return {mod(imagePoint.x, this->width), mod(imagePoint.y, this->height)};
}