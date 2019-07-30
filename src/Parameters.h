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
    float       sigma           = 1.f;
    std::size_t numberOfSteps   = 1000;
    float       tracerRadius      = 0.f;
    std::string imageFile       = "";

    Parameters() = default;
    Parameters(std::istream &input);

    void print(std::ostream &out);
};

#endif /* PARAMETERS_H_ */
