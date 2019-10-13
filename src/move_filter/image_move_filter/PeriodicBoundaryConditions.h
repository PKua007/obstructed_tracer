/*
 * PeriodicBoundaryConditions.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef PERIODICBOUNDARYCONDITIONS_H_
#define PERIODICBOUNDARYCONDITIONS_H_

#include "ImageBoundaryConditions.h"

/**
 * @brief `__host__ __device__` periodic ImageBoundaryConditions.
 */
class PeriodicBoundaryConditions : public ImageBoundaryConditions {
private:
    size_t width{};
    size_t height{};

    CUDA_HOSTDEV int mod(int a, int b) const { return (a % b + b) % b; }

public:
    CUDA_HOSTDEV void setupDimensions(size_t width, size_t height) override;

    /**
     * @brief All points are accepted due to infinite repeating of periodic boundary conditions - return always true.
     */
    CUDA_HOSTDEV bool isImagePointInBounds(ImagePoint imagePoint, int radius) const override;

    /**
     * @brief @a image point is translated according to periodic boundary conditions, so it ends up being in the range
     * of the image. After translating the result is returned.
     * @param imagePoint point to be translated
     * @return the point after translation
     */
    CUDA_HOSTDEV ImagePoint applyOnImagePoint(ImagePoint imagePoint) const override;
};

#endif /* PERIODICBOUNDARYCONDITIONS_H_ */
