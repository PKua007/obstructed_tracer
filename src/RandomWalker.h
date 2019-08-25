/*
 * RandomWalker.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef RANDOMWALKER_H_
#define RANDOMWALKER_H_

#include "Trajectory.h"

class RandomWalker {
public:
    virtual ~RandomWalker() = default;

    virtual void run(std::ostream &logger) = 0;
    virtual std::size_t getNumberOfTrajectories() const = 0;
    virtual const Trajectory &getTrajectory(std::size_t index) const = 0;
};

#endif /* RANDOMWALKER_H_ */
