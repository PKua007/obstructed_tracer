/*
 * ImageBoundaryConditions.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef IMAGEBOUNDARYCONDITIONS_H_
#define IMAGEBOUNDARYCONDITIONS_H_

#include "ImagePoint.h"

/**
 * @brief The `__host__` `__device__` boundary conditions on the edge of the image in ImageMoveFilter.
 */
class ImageBoundaryConditions {
public:
    CUDA_HOSTDEV virtual ~ImageBoundaryConditions() { };

    /**
     * @brief It setups the class for images sized @a width x @a height.
     *
     * @param width width of the image with boundary conditions
     * @param height height of the image with boundary conditions
     */
    CUDA_HOSTDEV virtual void setupDimensions(size_t width, size_t height) = 0;

    /**
     * @brief Returns true, if point is in bounds according to boundary conditions.
     * @param imagePoint point to be checked
     * @param radius radius of the tracer placed in @a imagePoint
     * @return true, if point is in bounds according to boundary conditions
     */
    CUDA_HOSTDEV virtual bool isImagePointInBounds(ImagePoint imagePoint, int radius) const = 0;

    /**
     * @brief Applies appropriate translation according to boundary conditions so that point is in the range of the
     * image and resurns the result.
     * @param imagePoint point to translate
     * @return the result of translation
     */
    CUDA_HOSTDEV virtual ImagePoint applyOnImagePoint(ImagePoint imagePoint) const = 0;
};

#endif /* IMAGEBOUNDARYCONDITIONS_H_ */
