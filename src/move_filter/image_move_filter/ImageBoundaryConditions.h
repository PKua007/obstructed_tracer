/*
 * ImageBoundaryConditions.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef IMAGEBOUNDARYCONDITIONS_H_
#define IMAGEBOUNDARYCONDITIONS_H_

#include "ImagePoint.h"

class ImageBoundaryConditions {
public:
    CUDA_HOSTDEV virtual ~ImageBoundaryConditions() = default;

    CUDA_HOSTDEV virtual void setupDimensions(size_t width, size_t height) = 0;
    CUDA_HOSTDEV virtual bool isImagePointInBounds(ImagePoint imagePoint, int radius) const = 0;
    CUDA_HOSTDEV virtual ImagePoint applyOnImagePoint(ImagePoint imagePoint) const = 0;
};

#endif /* IMAGEBOUNDARYCONDITIONS_H_ */
