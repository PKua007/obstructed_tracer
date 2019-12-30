/*
 * MoveFilterMock.h
 *
 *  Created on: 22 gru 2019
 *      Author: pkua
 */

#ifndef MOVEFILTERMOCK_H_
#define MOVEFILTERMOCK_H_

#include "trompeloeil_for_cuda/catch2/trompeloeil.hpp"

#include "simulation/MoveFilter.h"

class MoveFilterMock : public MoveFilter {
public:
    MAKE_CONST_MOCK2(isMoveValid, bool(Tracer tracer, Move move), override);
    MAKE_MOCK1(setupForTracerRadius, void(float radius), override);
    MAKE_MOCK0(randomValidTracer, Tracer(), override);
};

#endif /* MOVEFILTERMOCK_H_ */
