/*
 * Move.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

/** @file */

#ifndef MOVE_H_
#define MOVE_H_

#include "utils/CudaDefines.h"

/**
 * @brief A simple struct representing diffusion moves
 * @see operator+(Move, Move)
 */
struct Move {
    float x;
    float y;
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

#endif /* MOVE_H_ */
