/*
 * ImagePoint.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef IMAGEPOINT_H_
#define IMAGEPOINT_H_

#include "Point.h"

/**
 * @brief A `__host__` `__device__` struct similar to Point, but with integer coordinates.
 */
struct IntegerPoint {
    int x{};
    int y{};

    CUDA_HOSTDEV IntegerPoint() { };
    CUDA_HOSTDEV IntegerPoint(int x, int y) : x{x}, y{y} { };

    /**
     * @brief Construct IntegerPoint from @a point TRUNCATING the floats to int.
     */
    CUDA_HOSTDEV IntegerPoint(Point point) : x{static_cast<int>(point.x)}, y{static_cast<int>(point.y)} { };

    CUDA_HOSTDEV bool operator==(IntegerPoint second) const { return this->x == second.x && this->y == second.y; }
    CUDA_HOSTDEV bool operator!=(IntegerPoint second) const { return !(*this == second); }
};

#endif /* IMAGEPOINT_H_ */
