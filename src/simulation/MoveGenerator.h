/*
 * MoveGenerator.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef MOVEGENERATOR_H_
#define MOVEGENERATOR_H_

#include "Move.h"

/**
 * @brief This `__host__ __device__` interface generates random moves based on some distribution.
 */
class MoveGenerator {
public:
    CUDA_HOSTDEV virtual ~MoveGenerator() { };

    /**
     * @brief Generate random move based on some distribution.
     * @return random move based on some distribution
     */
    CUDA_HOSTDEV virtual Move generateMove() = 0;
};

#endif /* MOVEGENERATOR_H_ */
