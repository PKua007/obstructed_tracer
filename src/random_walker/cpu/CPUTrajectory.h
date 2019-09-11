/*
 * CPUTrajectory.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef CPUTRAJECTORY_H_
#define CPUTRAJECTORY_H_

#include <vector>

#include "../TrajectoryBase.h"

class CPUTrajectory : public TrajectoryBase {
public:
    CPUTrajectory() = default;
    CPUTrajectory(std::size_t numberOfPoints);

    void stayStill();
    void moveToPoint(Point point);
};

#endif /* CPUTRAJECTORY_H_ */
