/*
 * AccumulatingMSDDataCalculator.h
 *
 *  Created on: 25 sie 2019
 *      Author: pkua
 */

#ifndef ACCUMULATINGMSDDATACALCULATOR_H_
#define ACCUMULATINGMSDDATACALCULATOR_H_

#include <iosfwd>
#include <vector>

#include "MSDData.h"
#include "RandomWalker.h"

class AccumulatingMSDDataCalculator {
private:
    std::size_t numberOfTrajectories{};
    MSDData data;

public:
    AccumulatingMSDDataCalculator(std::size_t numberOfSteps) : data(numberOfSteps) { }

    void addTrajectories(const RandomWalker &randomWalker);
    MSDData fetchMSDData();
};


#endif /* ACCUMULATINGMSDDATACALCULATOR_H_ */
