/*
 * PeriodicBoundaryConditions.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef PERIODICBOUNDARYCONDITIONS_H_
#define PERIODICBOUNDARYCONDITIONS_H_

#include "utils/CudaDefines.h"

/**
 * @brief `__host__ __device__` periodic ImageBoundaryConditions.
 */
class PeriodicBoundaryConditions {
private:
    size_t width{};
    size_t height{};

    CUDA_HOSTDEV int mod(int a, int b) const { return (a % b + b) % b; }

public:
    CUDA_HOSTDEV PeriodicBoundaryConditions(size_t width, size_t height) : width{width}, height{height}
    { }

    /**
     * @brief All points are accepted due to infinite repeating of periodic boundary conditions - return always true.
     */
    CUDA_HOSTDEV bool isIntegerPointInBounds(IntegerPoint integerPoint, int radius) const {
        return true;
    }

    /**
     * @brief @a integerPoint is translated according to periodic boundary conditions, so it ends up being in the range
     * of the image. After translating the result is returned.
     * @param integerPoint point to be translated
     * @return the point after translation
     */
    CUDA_HOSTDEV IntegerPoint applyOnIntegerPoint(IntegerPoint integerPoint) const {
        return {mod(integerPoint.x, this->width), mod(integerPoint.y, this->height)};
    }
};

#endif /* PERIODICBOUNDARYCONDITIONS_H_ */
