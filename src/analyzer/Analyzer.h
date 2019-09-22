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

class Analyzer {
private:
    Parameters parameters;

public:
    struct Result {
        Quantity D{};
        Quantity alpha{};
    };

    Analyzer(const Parameters &parameters) : parameters(parameters) { }

    Result analyze(const MSDData &msdData);
};

#endif /* ANALYZER_H_ */
