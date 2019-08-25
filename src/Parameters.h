/*
 * Parameters.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <cstddef>
#include <iosfwd>


class Parameters {
private:
    void validateParameters() const;

public:
    std::size_t numberOfSteps = 1000;
    float       tracerRadius  = 0.f;
    std::string moveGenerator = "GaussianMoveGenerator";
    std::string moveFilter    = "DefaultMoveFilter";
    float       driftX        = 0.f;
    float       driftY        = 0.f;
    std::size_t numberOfWalks = 10;
    bool        storeTrajectories = false;
    std::string seed = "random";

    Parameters() = default;
    Parameters(std::istream &input);

    void print(std::ostream &out);
};

#endif /* PARAMETERS_H_ */
