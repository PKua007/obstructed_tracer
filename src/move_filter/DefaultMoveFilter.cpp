/*
 * DefaultMoveFilter.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include "DefaultMoveFilter.h"

bool DefaultMoveFilter::isMoveValid(Point tracer, Move move) const {
    return true;
}

Point DefaultMoveFilter::randomValidPoint() const {
    return {0.f, 0.f};
}
