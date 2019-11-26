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
#include "utils/Assertions.h"

/**
 * @brief A class calculating D and &alpha; from the simulations performed eariler.
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

private:
    Parameters parameters;
    double relativeRangeStart{};
    double relativeRangeEnd{};
    Result rSquareResult;
    Result rVarianceResult;
    double lastPointCorrelation{};
    double middlePointCorrelation{};

    double calculateCorrelation(MSDData::Entry msdEntry);

public:
    /**
     * @brief Creates analyzer for simulations performed using @a parameters.
     *
     * For other simulations it may not work properly.
     *
     * @param parameters the parameters used to perform simulation which will be analyzed
     * @param relativeRangeStart start of the fitting range given as fraction of total number of points, so from
     * [0, 1] interval
     * @param relativeRangeEnd start of the fitting range given as fraction of total number of points, so from
     * [0, 1] interval
     */
    Analyzer(const Parameters &parameters, double relativeRangeStart, double relativeRangeEnd)
            : parameters(parameters), relativeRangeStart{relativeRangeStart}, relativeRangeEnd{relativeRangeEnd}
    {
        Expects(relativeRangeStart >= 0. && relativeRangeStart <= 1.);
        Expects(relativeRangeEnd >= 0. && relativeRangeEnd <= 1.);
        Expects(relativeRangeStart < relativeRangeEnd);
    }

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
    void analyze(const MSDData &msdData);

    /**
     * @brief Returns the result of &lt;r<sup>2</sup>&gt;(t) fit.
     * @return the result of &lt;r<sup>2</sup>&gt;(t) fit
     */
    const Result& getRSquareResult() const {
        return rSquareResult;
    }

    /**
     * @brief Returns the result of &lt;var(x)+var(y)&gt;(t) fit.
     * @return the result of &lt;var(x)+var(y)&gt;(t) fit
     */
    const Result& getRVarianceResult() const {
        return rVarianceResult;
    }

    /**
     * @brief Returns the correlation cov(x, y)/&radic;(var(x) var(y)) for the last point in MSD data.
     * @return the correlation cov(x, y)/&radic;(var(x) var(y)) for the last point in MSD data
     */
    double getLastPointCorrelation() const {
        return lastPointCorrelation;
    }

    /**
     * @brief Returns the correlation cov(x, y)/&radic;(var(x) var(y)) for the middle point in log scale in MSD data.
     * @return the correlation cov(x, y)/&radic;(var(x) var(y)) forthe middle point in log scale in MSD data
     */
    double getMiddlePointCorrelation() const {
        return middlePointCorrelation;
    }
};

#endif /* ANALYZER_H_ */
