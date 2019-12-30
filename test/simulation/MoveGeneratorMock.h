/*
 * MoveGeneratorMock.h
 *
 *  Created on: 22 gru 2019
 *      Author: pkua
 */

#ifndef MOVEGENERATORMOCK_H_
#define MOVEGENERATORMOCK_H_

#include "trompeloeil_for_cuda/catch2/trompeloeil.hpp"

#include "simulation/MoveGenerator.h"

class MoveGeneratorMock : public MoveGenerator {
    MAKE_MOCK0(generateMove, Move(), override);
};

#endif /* MOVEGENERATORMOCK_H_ */
