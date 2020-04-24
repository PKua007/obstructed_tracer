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
     * @brief Returns true, if all pixels of a disk placed in @a integerPoint are in the bounds of the image.
     * @param integerPoint center of a tracer to check
     * @param radius of the tracer placed in @a integerPoint
     */
    CUDA_HOSTDEV bool isIntegerPointInBounds(IntegerPoint integerPoint, int radius) const {
        if (integerPoint.x - radius < 0 || integerPoint.x + radius >= this->width)
            return false;
        else if (integerPoint.y - radius < 0 || integerPoint.y + radius >= this->height)
            return false;
        else
            return true;
    }

    /**
     * @brief Performs no translation. The same point is returned
     * @return the same point as @a integerPoint
     */
    CUDA_HOSTDEV IntegerPoint applyOnIntegerPoint(IntegerPoint integerPoint) const {
        return integerPoint;
    }
};

#endif /* WALLBOUNDARYCONDITIONS_H_ */
