/*
 * MoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef MOVEFILTER_H_
#define MOVEFILTER_H_

#include "simulation/Tracer.h"

/**
 * @brief This `__host__ __device__` interface filters the moves of the tracer.
 *
 * <p> After setting up the tracer radius, the filter decides based on current tracer position and a move whether it
 * can be done. It can also provide random starting tracer which is valid from the point of this filter.
 *
 * <p>The canonical use of this class is to use MoveFilter::setupForTracer radius to specify the tracer radius, then
 * get random tracer by MoveFilter::randomValidTracer and use it in MoveFilter::isMoveValid. It will ensure that
 * everything is consistent and the class will always assume proper tracer radius.
 */
class MoveFilter {
public:
    CUDA_HOSTDEV virtual ~MoveFilter() { };

    /**
     * @brief Check whether the move is valid.
     *
     * If radius has not been set using setupForTracerRadius, the bahaviour is undefined. Otherwise it uses one chosen
     * there.
     *
     * @param tracer current tracer position
     * @param move move to perform
     * @return true if move is valid, false otherwisa
     */
    CUDA_HOSTDEV virtual bool isMoveValid(Tracer tracer, Move move) const = 0;

    /**
     * @brief Setups the MoveFilter for tracer radius @a radius.
     *
     * The radius is set globally, not as a parameter of isMoveValid and randomValidTracer methods for performance
     * reasons in the implementations - they can precompute some things.
     *
     * @param radius the radius of a tracer to be used
     */
    CUDA_HOSTDEV virtual void setupForTracerRadius(float radius) = 0;

    /**
     * @brief Returns a tracer for which `isMoveValid(tracer, 0) == true`.
     *
     * @return a valid tracer. The tracer radius is the one passed to MoveFilter::setupForTracerRadius
     */
    CUDA_HOSTDEV virtual Tracer randomValidTracer() = 0;
};

#endif /* MOVEFILTER_H_ */
