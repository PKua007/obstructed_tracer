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

class GPUTrajectory: public TrajectoryBase {
public:
    void moveGPUData(Point *gpuData, std::size_t size, std::size_t acceptedSteps);
};

#endif /* GPUTRAJECTORY_H_ */
