/*
 * PowerRegression.h
 *
 *  Created on: 22.06.2017
 *      Author: ciesla, pkua
 */

#ifndef POWERREGRESSION_H_
#define POWERREGRESSION_H_

#include <vector>
#include <limits>

#include "Quantity.h"

class PowerRegression {
    struct DataElement {
        double x{};
        double y{};

        DataElement() { }
        DataElement(double x, double y) : x{x}, y{y} { }
    };

private:
    static constexpr double UNDEFINED = std::numeric_limits<double>::quiet_NaN();

    std::vector<DataElement> data;

    double A = UNDEFINED;
    double lnB = UNDEFINED;
    double sigma2_A = UNDEFINED;
    double sigma2_lnB = UNDEFINED;
    double R2 = UNDEFINED;

public:
    void clear();

    /**
     * Add data point
     * @param x
     * @param y
     */
    void addXY(double x, double y);

    /**
     * calculates fit values
     */
    void calculate(int from, int to);

    /**
     * calculates fit values
     */
    void calculate();

    /**
     * @return parameter A from y = Bx^A with standard error
     */
    Quantity getExponent() const;

    /**
     * @return parameter B from y = Bx^A with standard error
     */
    Quantity getMultiplier() const;

    /**
     * @return R-squared coefficient
     */
    double getR2() const;

    int size() const;
};

#endif /* ANALIZATOR_LINEARREGRESSION_H_ */
