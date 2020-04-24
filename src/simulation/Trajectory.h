/*
 * Trajectory.h
 *
 *  Created on: 30 sie 2019
 *      Author: pkua
 */

#ifndef TRAJECTORY_H_
#define TRAJECTORY_H_

#include <iosfwd>
#include <vector>

#include "Point.h"

/**
 * @brief Class representing trajectory with additional routines to fetch data from gpu.
 *
 * Points can be added manually using TrajectoryImpl::addPoint, absorbed from another trajectory using
 * TrajectoryImpl::appendAnotherTrajectory or copied from GPU using TrajectoryImpl::copyGPUData.
 */
class Trajectory {
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
    using const_iterator = std::vector<Point>::const_iterator;

    /**
     * @brief Creates an empty trajectory.
     */
    Trajectory() = default;

    /**
     * @brief Creates an empty trajectory, but memory is reserved for @a numberOfPoints + 1 for initial tracer
     */
    Trajectory(std::size_t numberOfPoints);

    /**
     * @brief Returns the size of the trajectory INCLUDING the initial tracer position.
     * @return the size of the trajectory INCLUDING the initial tracer position
     */
    std::size_t getSize() const;

    /**
     * @brief Return the number of steps accepted by MoveFilter.
     * @return the number of steps accepted by MoveFilter
     */
    std::size_t getNumberOfAcceptedSteps() const;

    /**
     * @brief Returns the position of a tracer at step @a index.
     * @param index index of a position to return
     * @return the position of a tracer at step @a index
     */
    Point operator[](std::size_t index) const;

    /**
     * @brief Returns the initial position of a tracer
     * @return the initial position of a tracer
     */
    Point getFirst() const;

    /**
     * @brief Returns the final position of a tracer
     * @return the final position of a tracer
     */
    Point getLast() const;

    const_iterator begin() const { return this->trajectory.begin(); }
    const_iterator end() const { return this->trajectory.end(); }

    /**
     * @brief Clears the whole trajectory and resets the number of accepted steps.
     */
    void clear();

    /**
     * @brief Stores the trajectory in the output stream @a out in a text form.
     *
     * Each line contains one step in format "x y"
     *
     * @param out output stream to store the trajectory
     */
    void store(std::ostream &out) const;

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

#endif /* TRAJECTORY_H_ */
