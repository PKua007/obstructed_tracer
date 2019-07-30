/*
 * DefaultMoveFilter.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include "DefaultMoveFilter.h"

bool DefaultMoveFilter::isMoveValid(Tracer tracer, Move move) const {
    return true;
}

Tracer DefaultMoveFilter::randomValidTracer(float radius) {
    return Tracer({0.f, 0.f}, radius);
}
