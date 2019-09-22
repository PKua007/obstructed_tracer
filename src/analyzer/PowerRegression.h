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

class PowerRegression {
    struct DataElement {
        float x{};
        float y{};

        DataElement() { }
        DataElement(float x, float y) : x{x}, y{y} { }
    };

private:
    std::vector<DataElement> data;

    float A = std::numeric_limits<float>::quiet_NaN();
    float lnB = std::numeric_limits<float>::quiet_NaN();
    float sigma2_A = std::numeric_limits<float>::quiet_NaN();
    float sigma2_lnB = std::numeric_limits<float>::quiet_NaN();

public:
    void clear();

    /**
     * Add data point
     * @param x
     * @param y
     */
    void addXY(float x, float y);

    /**
     * calculates fit values
     */
    void calculate(int from, int to);

    /**
     * calculates fit values
     */
    void calculate();

    /**
     * @return parameter A from y = Bx^A
     */
    float getA();

    /**
     * @return standard deviation squared for parameter A fromy = Bx^A
     */
    float getSA();

    /**
     * @return parameter B from y = Bx^A
     */
    float getB();

    /**
     * @return standard deviation squared for parameter A fromy = Bx^A
     */
    float getSB();

    int size();
};

#endif /* ANALIZATOR_LINEARREGRESSION_H_ */
