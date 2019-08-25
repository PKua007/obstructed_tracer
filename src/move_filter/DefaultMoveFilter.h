/*
 * DefaultMoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef DEFAULTMOVEFILTER_H_
#define DEFAULTMOVEFILTER_H_

#include "../simulation/MoveFilter.h"

class DefaultMoveFilter : public MoveFilter {
public:
    bool isMoveValid(Tracer tracer, Move move) const override;
    Tracer randomValidTracer(float radius) override;
};

#endif /* DEFAULTMOVEFILTER_H_ */
