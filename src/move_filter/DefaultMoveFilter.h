/*
 * DefaultMoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef DEFAULTMOVEFILTER_H_
#define DEFAULTMOVEFILTER_H_

#include "simulation/MoveFilter.h"

class DefaultMoveFilter : public MoveFilter {
public:
    CUDA_HOSTDEV bool isMoveValid(Tracer tracer, Move move) const override;
    CUDA_HOSTDEV Tracer randomValidTracer(float radius) override;
};

#endif /* DEFAULTMOVEFILTER_H_ */
