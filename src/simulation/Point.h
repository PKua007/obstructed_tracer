/*
 * Point.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef POINT_H_
#define POINT_H_

#include <ostream>

#include "Move.h"
#include "utils/CudaDefines.h"

struct Point  {
    float x{};
    float y{};

    CUDA_HOSTDEV Point() { };
    CUDA_HOSTDEV Point(float x, float y) : x{x}, y{y} { }

    CUDA_HOSTDEV Point operator+(Move move) const { return {this->x + move.x, this->y + move.y}; }
    CUDA_HOSTDEV Point operator-(Move move) const { return {this->x - move.x, this->y - move.y}; }

    CUDA_HOSTDEV Point& operator+=(Move move) {
        this->x += move.x;
        this->y += move.y;
        return *this;
    }

    CUDA_HOSTDEV Point& operator-=(Move move) {
        this->x -= move.x;
        this->y -= move.y;
        return *this;
    }
};

inline std::ostream &operator<<(std::ostream& out, Point point) {
    out << point.x << " " << point.y;
    return out;
}

#endif /* POINT_H_ */
