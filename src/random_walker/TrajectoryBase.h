/*
 * TrajectoryBase.h
 *
 *  Created on: 30 sie 2019
 *      Author: pkua
 */

#ifndef TRAJECTORYBASE_H_
#define TRAJECTORYBASE_H_

#include <iterator>

#include "simulation/Trajectory.h"

/**
 * @brief A base class for CPU and GPU trajectories implementing the Trajectory interface.
 *
 * It gives access for child classes to internal TrajectoryBase::trajectory and TrajectoryBase::acceptedSteps fields
 * to set them anyway they want while taking care of the other boring stuff.
 */
class TrajectoryBase : public Trajectory {
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
    std::size_t getSize() const override { return this->trajectory.size(); }
    std::size_t getNumberOfAcceptedSteps() const override { return this->acceptedSteps; }
    Point operator[](std::size_t index) const override { return this->trajectory[index]; }
    Point getFirst() const override { return this->trajectory.front(); }
    Point getLast() const override{ return this->trajectory.back(); }

    void clear() override {
        this->trajectory.clear();
        this->acceptedSteps = 0;
    }

    void store(std::ostream &out) const override {
        std::copy(this->trajectory.begin(), this->trajectory.end(), std::ostream_iterator<Point>(out, "\n"));
    }
};

#endif /* TRAJECTORYBASE_H_ */
