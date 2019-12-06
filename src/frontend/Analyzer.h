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

/**
 * @brief An interface of class calculating D and &alpha; from the simulations performed eariler.
 *
 * It performs power fit y = Dx<sup>&alpha;</sup> to last two orders of point (namely [x<sub>max</sub>/100,
 * x<sub>max</sub>] range). The instance is created for specific Parameters and the incoming MSDData is expected to have
 * been generated using the same parameters.
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
     * More precisely, it makes y = Dx<sup>&alpha;</sup> fit to x<sub>max</sub>/100, x<sub>max</sub>] range. @a msdData
     * should be produced by a simulation which used Parameters passed in the constructor. The result can be obtained
     * using analyzer
     *
     * @param msdData mean square displacement data to be analyzed
     */
    virtual void analyze(const MSDData &msdData) = 0;

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
