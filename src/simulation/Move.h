/*
 * Move.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

/** @file */

#ifndef MOVE_H_
#define MOVE_H_

#include <ostream>

#include "utils/CudaDefines.h"

/**
 * @brief A simple struct representing diffusion moves
 * @see operator+(Move, Move)
 * @see operator*(Move, float)
 * @see operator*(float, Move)
 */
struct Move {
    float x{};
    float y{};

    CUDA_HOSTDEV Move() { };
    CUDA_HOSTDEV Move(float x, float y) : x{x}, y{y} { }

    CUDA_HOSTDEV Move &operator+=(Move other) {
        this->x += other.x;
        this->y += other.y;
        return *this;
    }

    CUDA_HOSTDEV Move &operator/=(float a) {
        this->x /= a;
        this->y /= a;
        return *this;
    }

    CUDA_HOSTDEV friend Move operator+(Move m1, Move m2) {
        return {m1.x + m2.x, m1.y + m2.y};
    }

    CUDA_HOSTDEV friend Move operator*(float s, Move m) {
        return {m.x * s, m.y * s};
    }

    CUDA_HOSTDEV friend Move operator*(Move m, float s) {
        return s*m;
    }

    CUDA_HOSTDEV friend Move operator/(Move m, float s) {
        return {m.x / s, m.y / s};
    }

    CUDA_HOSTDEV friend bool operator==(Move m1, Move m2) {
        return m1.x == m2.x && m1.y == m2.y;
    }

    CUDA_HOSTDEV friend bool operator!=(Move m1, Move m2) {
        return !(m1 == m2);
    }

    /**
     * @brief Stream insertion operator printing move in form "x y"
     */
    friend std::ostream &operator<<(std::ostream &out, Move move) {
        out << move.x << " " << move.y;
        return out;
    }
};

#endif /* MOVE_H_ */
