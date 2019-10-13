/*
 * GPUTrajectory.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef GPUTRAJECTORY_H_
#define GPUTRAJECTORY_H_

#include <vector>

#include "../TrajectoryBase.h"

/**
 * @brief Trajectory on GPU. The data is copied from GPU memory and number of accepted steps it given in the
 * constructor.
 */
class GPUTrajectory: public TrajectoryBase {
public:
    /**
     * @brief Created the trajectory in RAM by copying the data from @a gpuData GPU array.
     *
     * @param gpuData GPU array of trajectory points
     * @param size size of the trajectory INCLUDING initial tracer position
     * @param acceptedSteps the number of steps which has been accepted
     */
    void copyGPUData(Point *gpuData, std::size_t size, std::size_t acceptedSteps);
};

#endif /* GPUTRAJECTORY_H_ */
