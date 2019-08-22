/*
 * ImageBoundaryConditions.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef IMAGEBOUNDARYCONDITIONS_H_
#define IMAGEBOUNDARYCONDITIONS_H_

#include "ImagePoint.h"
#include "../../image/Image.h"

class ImageBoundaryConditions {
public:
    virtual ~ImageBoundaryConditions() = default;

    virtual void installOnImage(const Image &image) = 0;
    virtual bool isImagePointInBounds(ImagePoint imagePoint, int radius) const = 0;
    virtual ImagePoint applyOnImagePoint(ImagePoint imagePoint) const = 0;
};

#endif /* IMAGEBOUNDARYCONDITIONS_H_ */
