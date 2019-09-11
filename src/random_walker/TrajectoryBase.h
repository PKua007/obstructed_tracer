/*
 * TrajectoryBase.h
 *
 *  Created on: 30 sie 2019
 *      Author: pkua
 */

#ifndef TRAJECTORYBASE_H_
#define TRAJECTORYBASE_H_

#include <iterator>

#include "Trajectory.h"

class TrajectoryBase : public Trajectory {
protected:
    std::vector<Point> trajectory;
    std::size_t acceptedSteps{};

public:
    std::size_t getSize() const override { return this->trajectory.size(); }
    std::size_t getNumberOfAcceptedSteps() const override { return this->acceptedSteps; }
    Point operator[](std::size_t index) const override { return this->trajectory[index]; }
    Point getFirst() const override { return this->trajectory.front(); }
    Point getLast() const override{ return this->trajectory.back(); }
    void store(std::ostream &out) const override {
        std::copy(this->trajectory.begin(), this->trajectory.end(), std::ostream_iterator<Point>(out, "\n"));
    }
};

#endif /* TRAJECTORYBASE_H_ */
