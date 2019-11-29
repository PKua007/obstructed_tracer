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

#include "utils/Quantity.h"

/**
 * @brief A class performing power fit to points.
 */
class PowerRegression {
private:
    struct DataElement {
        double x{};
        double y{};

        DataElement() { }
        DataElement(double x, double y) : x{x}, y{y} { }
    };

    static constexpr double UNDEFINED = std::numeric_limits<double>::quiet_NaN();

    std::vector<DataElement> data;

    double A = UNDEFINED;
    double lnB = UNDEFINED;
    double sigma2_A = UNDEFINED;
    double sigma2_lnB = UNDEFINED;
    double R2 = UNDEFINED;

public:
    /**
     * @brief Resets object's state - all points and results are deleted.
     */
    void clear();

    /**
     * @brief Appends data point.
     */
    void addXY(double x, double y);

    /**
     * @brief After having added point (PowerRegression::addXY), it can be used to calculate fit values.
     *
     * The fit is made to points with INDICES of points from range [@a from, @a to). They can be then fetched using
     * getters: PowerRegression::getMultiplier, PowerRegression::getExponent and PowerRegression::getR2.
     *
     * @param from the INDEX of starting fit point, inclusive
     * @param to the INDEX of final fit point, exclusive
     */
    void calculate(int from, int to);

    /**
     * @brief Performs fit to all points.
     *
     * See PowerRegression::calculate(int, int) for expaination.
     */
    void calculate();

    /**
     * @brief After invoking PowerRegression::calculate or PowerRegression::calculate(int, int) it returns parameter A
     * from y = Bx<sup>A</sup> with standard error.
     *
     * @return parameter A from y = Bx<sup>A</sup> with standard error
     */
    Quantity getExponent() const;

    /**
     * @brief After invoking PowerRegression::calculate or PowerRegression::calculate(int, int) it returns parameter B
     * from y = Bx<sup>A</sup> with standard error.
     *
     * @return parameter B from y = Bx<sup>A</sup> with standard error
     */
    Quantity getMultiplier() const;

    /**
     * @brief After invoking PowerRegression::calculate or PowerRegression::calculate(int, int) it returns
     * R<sup>2</sup> measure of the quality of the fit.
     *
     * @return R<sup>2</sup> measure of the quality of the fit
     */
    double getR2() const;

    /**
     * @brief Returns the number of point added.
     * @return the number of point added
     */
    int size() const;
};

#endif /* ANALIZATOR_LINEARREGRESSION_H_ */
