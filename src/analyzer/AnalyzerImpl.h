/*
 * AnalyzerImpl.h
 *
 *  Created on: 22 wrz 2019
 *      Author: pkua
 */

#ifndef ANALYZERIMPL_H_
#define ANALYZERIMPL_H_

#include "frontend/Analyzer.h"
#include "frontend/Parameters.h"
#include "frontend/MSDData.h"
#include "utils/Quantity.h"
#include "utils/Assertions.h"

/**
 * @brief The concrete implementation of Analyzer.
 */
class AnalyzerImpl : public Analyzer {
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
    AnalyzerImpl(const Parameters &parameters, double relativeRangeStart, double relativeRangeEnd)
           : parameters(parameters), relativeRangeStart{relativeRangeStart}, relativeRangeEnd{relativeRangeEnd}
    {
        Expects(relativeRangeStart >= 0. && relativeRangeStart <= 1.);
        Expects(relativeRangeEnd >= 0. && relativeRangeEnd <= 1.);
        Expects(relativeRangeStart < relativeRangeEnd);
    }

    void analyze(const MSDData &msdData) override;

    const Result& getRSquareResult() const override { return rSquareResult; }
    const Result& getRVarianceResult() const override { return rVarianceResult; }
    double getLastPointCorrelation() const override { return lastPointCorrelation; }
    double getMiddlePointCorrelation() const override { return middlePointCorrelation; }
};

#endif /* ANALYZERIMPL_H_ */
