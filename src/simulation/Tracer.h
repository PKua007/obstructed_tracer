/*
 * Tracer.h
 *
 *  Created on: 30 lip 2019
 *      Author: pkua
 */

#ifndef TRACER_H_
#define TRACER_H_

#include "Point.h"
#include "utils/CudaQualifiers.h"

class Tracer {
private:
    Point position{};
    float radius{};

public:
    CUDA_HOSTDEV Tracer() = default;
    CUDA_HOSTDEV Tracer(Point position, float radius) : position{position}, radius{radius} { }

    CUDA_HOSTDEV Point getPosition() const { return position; }
    CUDA_HOSTDEV void setPosition(Point position) { this->position = position; }
    CUDA_HOSTDEV float getRadius() const { return radius; }
    CUDA_HOSTDEV void setRadius(float radius) { this->radius = radius; }

    CUDA_HOSTDEV Tracer &operator+=(Move move) {
        this->position += move;
        return *this;
    }
};

#endif /* TRACER_H_ */
