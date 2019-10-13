/*
 * DefaultMoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef DEFAULTMOVEFILTER_H_
#define DEFAULTMOVEFILTER_H_

#include "random_walker/MoveFilter.h"

/**
 * @brief A `__host__` `__device__` filter accepting all moves.
 */
class DefaultMoveFilter : public MoveFilter {
private:
    float tracerRadius{};

public:
    /**
     * @brief Always returns `true`
     */
    CUDA_HOSTDEV bool isMoveValid(Tracer tracer, Move move) const override;

    /**
     * @brief Returns `Tracer{}`
     * @return `Tracer{}`
     */
    CUDA_HOSTDEV Tracer randomValidTracer() override;

    /**
     * @brief No setup is needed. Is accepts everything anyway.
     */
    CUDA_HOSTDEV void setupForTracerRadius(float radius) override;
};

#endif /* DEFAULTMOVEFILTER_H_ */
