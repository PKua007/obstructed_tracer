/*
 * GPURandomWalker.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef GPURANDOMWALKER_H_
#define GPURANDOMWALKER_H_

#include "RandomWalker.h"

class GPURandomWalker : public RandomWalker {
public:
    void run(std::ostream &logger) override;
    std::size_t getNumberOfTrajectories() const override;
    const Trajectory &getTrajectory(std::size_t index) const override;
};

#endif /* GPURANDOMWALKER_H_ */
