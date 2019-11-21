/*
 * WallBoundaryConditions.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef WALLBOUNDARYCONDITIONS_H_
#define WALLBOUNDARYCONDITIONS_H_

#include "utils/CudaDefines.h"

/**
 * @brief `__host__ __device__` wall ImageBoundaryConditions, ie. borders can't be passed.
 */
class WallBoundaryConditions {
private:
    std::size_t width{};
    std::size_t height{};

public:
    CUDA_HOSTDEV WallBoundaryConditions(size_t width, size_t height) : width{width}, height{height}
    { }

    /**
     * @brief Returns true, if all pixels of a disk placed in @a imagePoint are in the bounds of the image.
     * @param imagePoint center of a tracer to check
     * @param radius of the tracer placed in @a imagePoint
     */
    CUDA_HOSTDEV bool isImagePointInBounds(ImagePoint imagePoint, int radius) const {
        if (imagePoint.x - radius < 0 || imagePoint.x + radius >= this->width)
            return false;
        else if (imagePoint.y - radius < 0 || imagePoint.y + radius >= this->height)
            return false;
        else
            return true;
    }

    /**
     * @brief Performs no translation. The same point is returned
     * @return the same point as @a imagePoint
     */
    CUDA_HOSTDEV ImagePoint applyOnImagePoint(ImagePoint imagePoint) const {
        return imagePoint;
    }
};

#endif /* WALLBOUNDARYCONDITIONS_H_ */
