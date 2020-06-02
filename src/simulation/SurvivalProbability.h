/*
 * SurvivalProbability.h
 *
 *  Created on: 1 cze 2020
 *      Author: pkua
 */

#ifndef SURVIVALPROBABILITY_H_
#define SURVIVALPROBABILITY_H_

#include <vector>
#include <iosfwd>

#include "utils/Assertions.h"

/**
 * @brief Class storing survival probability, so for each time, the percentage of trajectories which never escaped
 * a circle of given radius up to this time.
 */
class SurvivalProbability {
private:
    double radius{};
    std::size_t stepSize{};
    double integrationStep{};
    std::vector<double> data;

public:
    using iterator = std::vector<double>::iterator;
    using const_iterator = std::vector<double>::const_iterator;

    /**
     * @brief Constructs the class
     * @param radius radius of a circle which should not be escaped from
     * @param numSteps number of steps, each of size @a stepSize. The memory will actually be reserved for @a numSteps
     * + 1, because t=0 is also included
     * @param integrationStep the integration step (time delta per trajectory step - not SP step)
     */
    SurvivalProbability(double radius, std::size_t numSteps, std::size_t stepSize, double integrationStep);

    double getRadius() const { return this->radius; }

    double operator[](std::size_t i) const {
        Expects(i < this->data.size());
        return this->data[i];
    }

    double &operator[](std::size_t i) {
        Expects(i < this->data.size());
        return this->data[i];
    }

    std::size_t size() const { return this->data.size(); }
    iterator begin() { return this->data.begin(); }
    const_iterator begin() const  { return this->data.begin(); }
    iterator end() { return this->data.end(); }
    const_iterator end() const { return this->data.end(); }

    void store(std::ostream &out) const;
};

#endif /* SURVIVALPROBABILITY_H_ */
