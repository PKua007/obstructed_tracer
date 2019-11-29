/*
 * TrajectoryImpl.h
 *
 *  Created on: 30 sie 2019
 *      Author: pkua
 */

#ifndef TRAJECTORYIMPL_H_
#define TRAJECTORYIMPL_H_

#include <iosfwd>
#include <vector>

#include "simulation/Trajectory.h"

/**
 * @brief Trajectory both for for CPU and GPU.
 *
 * Points can be added manually using TrajectoryImpl::addPoint, absorbed from another trajectory using
 * TrajectoryImpl::appendAnotherTrajectory or copied from GPU using TrajectoryImpl::copyGPUData.
 */
class TrajectoryImpl : public Trajectory {
protected:
    /**
     * @brief Internal vector of points to be modified by subclasses.
     */
    std::vector<Point> trajectory;

    /**
     * @brief Internal field with number of steps to be modified by subclasses.
     */
    std::size_t acceptedSteps{};

public:
    /**
     * @brief Creates an empty trajectory.
     */
    TrajectoryImpl() = default;

    /**
     * @brief Creates an empty trajectory, but memory is reserved for @a numberOfPoints + 1 for initial tracer
     */
    TrajectoryImpl(std::size_t numberOfPoints);

    std::size_t getSize() const override;
    std::size_t getNumberOfAcceptedSteps() const override;
    Point operator[](std::size_t index) const override;
    Point getFirst() const override;
    Point getLast() const override;
    void clear() override;
    void store(std::ostream &out) const override;

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

    /**
     * @brief Created the trajectory in RAM by copying the data from @a gpuData GPU array.
     *
     * @param gpuData GPU array of trajectory points
     * @param size size of the trajectory INCLUDING initial tracer position
     * @param acceptedSteps the number of steps which has been accepted
     */
    void copyGPUData(Point *gpuData, std::size_t size, std::size_t acceptedSteps);
};

#endif /* TRAJECTORYIMPL_H_ */
