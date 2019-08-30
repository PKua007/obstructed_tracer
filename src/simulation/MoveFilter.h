/*
 * MoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef MOVEFILTER_H_
#define MOVEFILTER_H_

#include "Tracer.h"

class MoveFilter {
public:
    CUDA_HOSTDEV virtual ~MoveFilter() = default;

    CUDA_HOSTDEV virtual bool isMoveValid(Tracer tracer, Move move) const = 0;
    CUDA_HOSTDEV virtual Tracer randomValidTracer(float radius) = 0;
};

#endif /* MOVEFILTER_H_ */
