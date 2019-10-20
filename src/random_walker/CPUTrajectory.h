/*
 * CPUTrajectory.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef CPUTRAJECTORY_H_
#define CPUTRAJECTORY_H_

#include <vector>

#include "TrajectoryBase.h"

/**
 * @brief Trajectory created by CPURandomWalker or SplitRandomWalker.
 */
class CPUTrajectory : public TrajectoryBase {
public:
    /**
     * @brief Creates an empty trajectory.
     */
    CPUTrajectory() = default;

    /**
     * @brief Creates an empty trajectory, but memory is reserved for @a numberOfPoints + 1 for initial tracer
     */
    CPUTrajectory(std::size_t numberOfPoints);

    /**
     * @brief Append a @a point to the trajectory.
     *
     * The first point cannot be counted as accepted.
     *
     * @param point point to be added
     * @param isAccepted if true, the accepted steps counter is increased
     */
    void addPoint(Point point, bool isAccepted = false);

    /**
     * @brief Appends another trajectory at the end of this trajectory.
     *
     * The first point of @a trajectory has to match the end of this trajectory.
     */
    void appendAnotherTrajectory(const Trajectory &trajectory);
};

#endif /* CPUTRAJECTORY_H_ */
