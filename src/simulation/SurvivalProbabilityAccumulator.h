/*
 * SurvivalProbabilityAccumulator.h
 *
 *  Created on: 1 cze 2020
 *      Author: pkua
 */

#ifndef SURVIVALPROBABILITYACCUMULATOR_H_
#define SURVIVALPROBABILITYACCUMULATOR_H_

#include "SurvivalProbability.h"
#include "Trajectory.h"

class SurvivalProbabilityAccumulator {
public:
    SurvivalProbabilityAccumulator(const std::vector<double> &radii, std::size_t numSteps, std::size_t stepDelta,
                                   double integrationStep);

    void addTrajectories(const std::vector<Trajectory> &trajectories);
    std::vector<SurvivalProbability> calculateSurvivalProbabilities() const;
};

#endif /* SURVIVALPROBABILITYACCUMULATOR_H_ */
