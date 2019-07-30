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

Point DefaultMoveFilter::randomValidPoint() {
    return {0.f, 0.f};
}
