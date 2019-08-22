/*
 * WallBoundaryConditions.cpp
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#include "WallBoundaryConditions.h"

void WallBoundaryConditions::installOnImage(const Image& image) {
    this->width = image.getWidth();
    this->height = image.getHeight();
}

bool WallBoundaryConditions::isImagePointInBounds(ImagePoint imagePoint, int radius) const {
    if (imagePoint.x - radius < 0 || imagePoint.x + radius >= this->width)
        return false;
    else if (imagePoint.y - radius < 0 || imagePoint.y + radius >= this->height)
        return false;
    else
        return true;
}

ImagePoint WallBoundaryConditions::applyOnImagePoint(ImagePoint imagePoint) const {
    return imagePoint;
}
