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

/**
 * @brief Trajectory created by CPURandomWalker.
 *
 * The subsequent points are added using CPUTrajectory::stayStill or CPUTrajectory::moveToPoint methods. This two
 * methods are also used to count the number of accepted steps - the use of stayStill does not increment the counter,
 * while moveToPoint does.
 */
class CPUTrajectory : public TrajectoryBase {
public:
    /**
     * @brief Created an empty trajectory.
     */
    CPUTrajectory() = default;

    /**
     * @brief Creates an empty trajectory, however the memory is reserved for @a numberOfPoint points INCLUDING initial
     * tracer position.
     *
     * @param numberOfPoints desired number of point in the trajectory EXLCUDING the initial tracer position
     * @param initialPosition initial position of a tracer; it will not be included in accepted steps
     */
    CPUTrajectory(std::size_t numberOfPoints, Point initialPosition);

    /**
     * @brief Copies the last point of the trajectory to next. The accepted steps counter is not incremented.
     */
    void stayStill();

    /**
     * @brief Append a @a point to the trajectory and inctrements accepted step counter.
     */
    void moveToPoint(Point point);

    void appendAnotherTrajectory(const Trajectory &trajectory);
};

#endif /* CPUTRAJECTORY_H_ */
