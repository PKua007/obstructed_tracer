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
};

/**
 * @brief Performs an addition on two Move objects @a m1 and @a m2.
 * @param m1 lhs operand
 * @param m2 rhs operand
 * @return the two moves added
 */
CUDA_HOSTDEV inline Move operator+(Move m1, Move m2) {
    return {m1.x + m2.x, m1.y + m2.y};
}

/**
 * @brief Performs an multiplication of Move object @a m by a scalar @a s
 * @param s scalar to multiply move with
 * @param m move to rescale
 * @return the move rescaled
 */
CUDA_HOSTDEV inline Move operator*(float s, Move m) {
    return {m.x * s, m.y * s};
}

/**
 * @brief Performs an multiplication of Move object @a m by a scalar @a s
 * @param m move to rescale
 * @param s scalar to multiply move with
 * @return the move rescaled
 */
CUDA_HOSTDEV inline Move operator*(Move m, float s) {
    return s*m;
}

/**
 * @brief Checks if @a m1 and @a m2 are equal.
 */
inline bool operator==(Move m1, Move m2) {
    return m1.x == m2.x && m1.y == m2.y;
}

/**
 * @brief Checks if @a m1 and @a m2 are not equal.
 */
inline bool operator!=(Move m1, Move m2) {
    return !(m1 == m2);
}

/**
 * @brief Stream insertion operator printing move in form "x y"
 * @param out stream to print the move to
 * @param move move to be printed
 * @return the reference to @a out
 */
inline std::ostream &operator<<(std::ostream &out, Move move) {
    out << move.x << " " << move.y;
    return out;
}

#endif /* MOVE_H_ */
