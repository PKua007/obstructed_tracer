/*
 * Trajectory.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef TRAJECTORY_H_
#define TRAJECTORY_H_

#include "Point.h"

/**
 * @brief A simple interface representing a trajectory.
 *
 * It gives basic immutable access methods and also storing the trajectory in the output stream. How the points are
 * added and how many steps were accepted is up to implementing classes.
 */
class Trajectory {
public:
    virtual ~Trajectory() = default;

    /**
     * @brief Returns the size of the trajectory INCLUDING the initial tracer position.
     * @return the size of the trajectory INCLUDING the initial tracer position
     */
    virtual std::size_t getSize() const = 0;

    /**
     * @brief Return the number of steps accepted by MoveFilter.
     * @return the number of steps accepted by MoveFilter
     */
    virtual std::size_t getNumberOfAcceptedSteps() const = 0;

    /**
     * @brief Returns the position of a tracer at step @a index.
     * @param index index of a position to return
     * @return the position of a tracer at step @a index
     */
    virtual Point operator[](std::size_t index) const = 0;

    /**
     * @brief Returns the initial position of a tracer
     * @return the initial position of a tracer
     */
    virtual Point getFirst() const = 0;

    /**
     * @brief Returns the final position of a tracer
     * @return the final position of a tracer
     */
    virtual Point getLast() const = 0;

    /**
     * @brief Clears the whole trajectory and resets the number of accepted steps.
     */
    virtual void clear() = 0;

    /**
     * @brief Stores the trajectory in the output stream @a out in a text form.
     *
     * Each line contains one step in format "x y"
     *
     * @param out output stream to store the trajectory
     */
    virtual void store(std::ostream &out) const = 0;
};

#endif /* TRAJECTORY_H_ */
