/*
 * DefaultMoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef DEFAULTMOVEFILTER_H_
#define DEFAULTMOVEFILTER_H_

#include "random_walker/MoveFilter.h"

class DefaultMoveFilter : public MoveFilter {
private:
    float tracerRadius{};

public:
    CUDA_HOSTDEV bool isMoveValid(Tracer tracer, Move move) const override;
    CUDA_HOSTDEV Tracer randomValidTracer() override;
    CUDA_HOSTDEV void setupForTracerRadius(float radius) override;
};

#endif /* DEFAULTMOVEFILTER_H_ */
