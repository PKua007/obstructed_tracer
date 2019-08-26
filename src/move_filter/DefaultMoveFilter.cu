/*
 * DefaultMoveFilter.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include "DefaultMoveFilter.h"

CUDA_HOSTDEV bool DefaultMoveFilter::isMoveValid(Tracer tracer, Move move) const {
    return true;
}

CUDA_HOSTDEV Tracer DefaultMoveFilter::randomValidTracer(float radius) {
    return Tracer({0.f, 0.f}, radius);
}
