/*
 * Tracer.h
 *
 *  Created on: 30 lip 2019
 *      Author: pkua
 */

#ifndef TRACER_H_
#define TRACER_H_

#include "simulation/Point.h"
#include "utils/CudaDefines.h"

/**
 * @brief A `__host__ __device__` class representing a finite-sized tracer.
 */
class Tracer {
private:
    Point position{};
    float radius{};

public:
    CUDA_HOSTDEV Tracer() { };
    CUDA_HOSTDEV Tracer(Point position, float radius) : position{position}, radius{radius} { }

    /**
     * @brief Replaces the old position of a tracer leaving the tracer radius unaltered.
     * @return the position of the tracer
     */
    CUDA_HOSTDEV Tracer &operator=(Point position) {
        this->position = position;
        return *this;
    }

    /**
     * @brief Returns the position of the tracer.
     * @return the position of the tracer
     */
    CUDA_HOSTDEV Point getPosition() const { return position; }

    /**
     * @brief Sets the new position of a tracer.
     * @param position the new position of a tracer
     */
    CUDA_HOSTDEV void setPosition(Point position) { this->position = position; }

    /**
     * @brief Returns the radius of the tracer.
     * @return the radius of the tracer
     */
    CUDA_HOSTDEV float getRadius() const { return radius; }

    /**
     * @brief Sets the new radius of a tracer.
     * @param radius the new radius of a tracer
     */
    CUDA_HOSTDEV void setRadius(float radius) { this->radius = radius; }

    /**
     * @brief Applies a move to the tracer.
     * @param move a move to be applied
     * @return a reference to this
     */
    CUDA_HOSTDEV Tracer &operator+=(Move move) {
        this->position += move;
        return *this;
    }

    friend CUDA_HOSTDEV bool operator==(const Tracer &, const Tracer &);
};

/**
 * @brief Compares two Tracer-s.
 */
inline CUDA_HOSTDEV bool operator==(const Tracer &t1, const Tracer &t2) {
    return t1.position == t2.position
           && t1.radius == t2.radius;
}

#endif /* TRACER_H_ */
