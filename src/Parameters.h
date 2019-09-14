/*
 * Parameters.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <cstddef>
#include <string>


class Parameters {
private:
    void validateParameters() const;

public:
    std::size_t numberOfSteps         = 1000;
    float       tracerRadius          = 0.f;
    std::string moveGenerator         = "GaussianMoveGenerator";
    std::string moveFilter            = "DefaultMoveFilter";
    std::string drift                 = "xy 0 0";
    std::size_t numberOfWalksInSeries = 10;
    std::size_t numberOfSeries        = 1;
    bool        storeTrajectories     = false;
    std::string seed                  = "random";
    std::string device                = "gpu";

    Parameters() = default;
    Parameters(std::istream &input);

    void print(std::ostream &out);
};

#endif /* PARAMETERS_H_ */
