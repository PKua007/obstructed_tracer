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
    Result rSquareResult;
    Result rVarianceResult;
    double lastPointCorrelation{};
    double middlePointCorrelation{};

    double calculateCorrelation(MSDData::Entry msdEntry);

public:
    void analyze(const MSDData &msdData, const Parameters &parameters, double relativeRangeStart,
                 double relativeRangeEnd) override;

    const Result &getRSquareResult() const override { return rSquareResult; }
    const Result &getRVarianceResult() const override { return rVarianceResult; }
    double getLastPointCorrelation() const override { return lastPointCorrelation; }
    double getMiddlePointCorrelation() const override { return middlePointCorrelation; }
};

#endif /* ANALYZERIMPL_H_ */
