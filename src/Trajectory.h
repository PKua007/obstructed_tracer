/*
 * Trajectory.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef TRAJECTORY_H_
#define TRAJECTORY_H_

#include "Point.h"

class Trajectory {
public:
    virtual ~Trajectory() = default;

    virtual std::size_t getSize() const = 0;
    virtual std::size_t getNumberOfAcceptedSteps() const = 0;
    virtual Point operator[](std::size_t index) const = 0;
    virtual Point getFirst() const = 0;
    virtual Point getLast() const = 0;
    virtual void store(std::ostream &out) const = 0;
};

#endif /* TRAJECTORY_H_ */
