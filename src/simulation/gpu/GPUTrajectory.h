/*
 * GPUTrajectory.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef GPUTRAJECTORY_H_
#define GPUTRAJECTORY_H_

#include "Trajectory.h"

class GPUTrajectory: public Trajectory {
public:
    std::size_t getSize() const override;
    std::size_t getNumberOfAcceptedSteps() const override;
    Point operator[](std::size_t index) const override;
    Point getFirst() const override;
    Point getLast() const override;

    void store(std::ostream &out) const override;
};

#endif /* GPUTRAJECTORY_H_ */
