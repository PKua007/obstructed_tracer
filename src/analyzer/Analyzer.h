/*
 * Analyzer.h
 *
 *  Created on: 22 wrz 2019
 *      Author: pkua
 */

#ifndef ANALYZER_H_
#define ANALYZER_H_

#include "Parameters.h"
#include "MSDData.h"
#include "Quantity.h"

/**
 * @brief A class calculating D and &alpha; from the simulations performed eariler.
 *
 * It performs power fit y = Dx<sup>&alpha;</sup> to last two orders of point (namely [x<sub>max</sub>/100,
 * x<sub>max</sub>] range). The instance is created for specific Parameters and the incoming MSDData is expected to have
 * been generated using the same parameters.
 */
class Analyzer {
private:
    Parameters parameters;

public:
    /**
     * @brief The results of the power fit y = Dx<sup>&alpha;</sup>.
     */
    struct Result {
        /**
         * @brief A diffusion coefficient.
         */
        Quantity D{};

        /**
         * @brief A diffusion exponent.
         */
        Quantity alpha{};

        /**
         * @brief R<sup>2</sup> measure of the quality of the fit.
         */
        double R2{};
    };

    /**
     * @brief Creates analyzer for simulations performed using @a parameters.
     *
     * For other simulations it may not work properly.
     *
     * @param parameters the parameters used to perform simulation which will be analyzed
     */
    Analyzer(const Parameters &parameters) : parameters(parameters) { }

    /**
     * @brief Performs the power fit to last two orders of points and returns the result.
     *
     * More precisely, it makes y = Dx<sup>&alpha;</sup> fit to x<sub>max</sub>/100, x<sub>max</sub>] range. @a msdData
     * should be produced by a simulation which used Parameters passed in the constructor.
     *
     * @param msdData mean square displacement data to be analyzed
     * @return the results of the fit with R<sup>2</sup>
     */
    Result analyze(const MSDData &msdData);
};

#endif /* ANALYZER_H_ */
