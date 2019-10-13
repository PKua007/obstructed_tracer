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

/**
 * @brief A struct containing all parameters of the simulation.
 *
 * It is read from input file. See input.txt for the meaning of the fields and the constructor to know how parameters
 * are parsed.
 */
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

    /**
     * @brief Creates default parameters.
     */
    Parameters() = default;

    /**
     * @brief Read the parameters from @a input stream using Config format.
     *
     * This is
     * <blockquote>key = value</blockquote>
     *
     * The keys have the same names as the public fields of the class. Not all of them have to be explicitly specified.
     * The remaining ones will then have the default values.
     *
     * @param input input stream to read parameters from
     */
    Parameters(std::istream &input);

    /**
     * @brief Prints info about parameters onto @a out stream.
     * @param out stream to print into about parameters to
     */
    void print(std::ostream &out);
};

#endif /* PARAMETERS_H_ */
