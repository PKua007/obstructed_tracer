/*
 * MoveGenerator.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef MOVEGENERATOR_H_
#define MOVEGENERATOR_H_

#include "Move.h"

class MoveGenerator {
public:
    CUDA_HOSTDEV virtual ~MoveGenerator() = default;

    CUDA_HOSTDEV virtual Move generateMove() = 0;
};

#endif /* MOVEGENERATOR_H_ */
