/*
 * DefaultMoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef DEFAULTMOVEFILTER_H_
#define DEFAULTMOVEFILTER_H_

#include "../random_walker/MoveFilter.h"

class DefaultMoveFilter : public MoveFilter {
public:
    bool isMoveValid(Point tracer, Move move) const;
    Point randomValidPoint();
};

#endif /* DEFAULTMOVEFILTER_H_ */
