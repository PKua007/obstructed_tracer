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
private:
    std::vector<double> radii;
    std::size_t numSteps{};
    std::size_t stepSize{};
    double integrationStep{};
    std::size_t numTrajectories{};

    std::vector<std::vector<double>> data;

public:
    SurvivalProbabilityAccumulator(const std::vector<double> &radii, std::size_t numSteps, std::size_t stepSize,
                                   double integrationStep);

    void addTrajectories(const std::vector<Trajectory> &trajectories);
    std::vector<SurvivalProbability> calculateSurvivalProbabilities() const;

    std::size_t getStepSize() const { return this->stepSize; }
    const std::vector<double> &getRadii() const { return this->radii; }
};

#endif /* SURVIVALPROBABILITYACCUMULATOR_H_ */
