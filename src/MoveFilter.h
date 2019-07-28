/*
 * MoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef MOVEFILTER_H_
#define MOVEFILTER_H_

#include "Point.h"
#include "Move.h"

class MoveFilter {
public:
    virtual ~MoveFilter() = default;

    virtual bool isMoveValid(Point tracer, Move move) const = 0;
    virtual Point randomValidPoint() const = 0;
};

#endif /* MOVEFILTER_H_ */
