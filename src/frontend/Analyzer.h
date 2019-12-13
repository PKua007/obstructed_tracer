/*
 * Analyzer.h
 *
 *  Created on: 29 lis 2019
 *      Author: pkua
 */

#ifndef ANALYZER_H_
#define ANALYZER_H_

#include "Analyzer.h"
#include "MSDData.h"
#include "utils/Quantity.h"
#include "Parameters.h"

/**
 * @brief An interface of class calculating D and &alpha; from the simulations performed eariler.
 *
 * It performs power fit y = Dx<sup>&alpha;</sup> to the specified range of times. The analyzis is made for specific
 * Parameters and the incoming MSDData is expected to have been generated using the same parameters.
 */
class Analyzer {
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
     * @brief Performs the power fit to last two orders of points &lt;r<sup>2</sup>&gt;(t) and
     * &lt;var(x)+var(y)&gt;(t).
     *
     * More precisely, it makes y = Dx<sup>&alpha;</sup> fit to [t<sub>min</sub>, t<sub>max</sub>] range, where
     * t<sub>min</sub> = t * @a relativeRangeStart, t<sub>max</sub> = t * @a relativeRangeEnd. @a msdData should be
     * produced by a simulation which used @a parameters. The result can be obtained using the getters. It also
     * calculates the correlation of x and y for last point from fit and middle point - it corresponds to a
     * &radic;t<sub>max</sub>.
     *
     * @param msdData mean square displacement data to be analyzed
     * @param parameters Parameters used to perform walks which are now being analyzed
     * @param relativeRangeStart a start of fit given in relative to max time (a number from (0, 1] interval)
     * @param relativeRangeEnd an end of fit given in relative to max time (a number from (0, 1] interval)
     */
    virtual void analyze(const MSDData &msdData, const Parameters &parameters, double relativeRangeStart,
                         double relativeRangeEnd) = 0;

    /**
     * @brief Returns the result of &lt;r<sup>2</sup>&gt;(t) fit.
     * @return the result of &lt;r<sup>2</sup>&gt;(t) fit
     */
    virtual const Result& getRSquareResult() const = 0;

    /**
     * @brief Returns the result of &lt;var(x)+var(y)&gt;(t) fit.
     * @return the result of &lt;var(x)+var(y)&gt;(t) fit
     */
    virtual const Result& getRVarianceResult() const = 0;

    /**
     * @brief Returns the correlation cov(x, y)/&radic;(var(x) var(y)) for the last point in MSD data.
     * @return the correlation cov(x, y)/&radic;(var(x) var(y)) for the last point in MSD data
     */
    virtual double getLastPointCorrelation() const = 0;

    /**
     * @brief Returns the correlation cov(x, y)/&radic;(var(x) var(y)) for the middle point in log scale in MSD data.
     * @return the correlation cov(x, y)/&radic;(var(x) var(y)) forthe middle point in log scale in MSD data
     */
    virtual double getMiddlePointCorrelation() const = 0;
};

#endif /* ANALYZER_H_ */
