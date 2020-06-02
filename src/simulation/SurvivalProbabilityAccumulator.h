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

/**
 * @brief Class which accumulates SurvivalProbability for different radii for all passed trajectories.
 */
class SurvivalProbabilityAccumulator {
private:
    std::vector<double> radii;
    std::size_t numSteps{};
    std::size_t stepSize{};
    double integrationStep{};
    std::size_t numTrajectories{};

    std::vector<std::vector<double>> data;

public:
    /**
     * @brief Constructs accumulator creating SurvivalProbability -ies for all given @a radii.
     * @details The rest of parameters are passed to SurvivalProbability - check its constructor.
     */
    SurvivalProbabilityAccumulator(const std::vector<double> &radii, std::size_t numSteps, std::size_t stepSize,
                                   double integrationStep);

    void addTrajectories(const std::vector<Trajectory> &trajectories);
    std::vector<SurvivalProbability> calculateSurvivalProbabilities() const;

    std::size_t getStepSize() const { return this->stepSize; }
    const std::vector<double> &getRadii() const { return this->radii; }
};

#endif /* SURVIVALPROBABILITYACCUMULATOR_H_ */
