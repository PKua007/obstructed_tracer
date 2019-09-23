/*
 * LinearRegression.cpp
 *
 *  Created on: 22.06.2017
 *      Author: ciesla, pkua
 */

#include "PowerRegression.h"
#include "utils/Assertions.h"

#include <algorithm>
#include <cmath>

void PowerRegression::clear() {
    (*this) = PowerRegression{}; // A brand new one - clear and with NaNs
}

void PowerRegression::addXY(float x, float y) {
    Expects(x > 0);
    Expects(y > 0);
    this->data.push_back({x, y});
}

void PowerRegression::calculate(int from, int to) {
    int n = to - from;
    Expects(n > 2);

    float sum_lnx{};
    float sum_lny{};
    float sum_lnxlny{};
    float sum_lnx2{};
    for (int i = from; i < to; i++) {
        DataElement d = this->data[i];
        float lnx = std::log(d.x);
        float lny = std::log(d.y);

        sum_lnxlny += lnx*lny;
        sum_lnx += lnx;
        sum_lny += lny;
        sum_lnx2 += lnx*lnx;
    }
    this->A = (n*sum_lnxlny - sum_lnx*sum_lny) / (n*sum_lnx2 - sum_lnx*sum_lnx);
    this->lnB = (sum_lny - this->A*sum_lnx)/n;

    float sigma2{};
    for (int i = from; i < to; i++) {
        DataElement d = this->data[i];
        sigma2 += std::pow(std::log(d.y) - this->lnB - this->A*std::log(d.x), 2);
    }
    float mse2 = sigma2/(n - 2);

    this->sigma2_A = (n/(n*sum_lnx2 - sum_lnx*sum_lnx))*mse2;

    // This is added by PKua based on https://www.cse.wustl.edu/~jain/iucee/ftp/k_14slr.pdf
    this->sigma2_lnB = sum_lnx2/(n*sum_lnx2 - sum_lnx*sum_lnx)*mse2;
}

void PowerRegression::calculate() {
    this->calculate(0, this->data.size());
}

Quantity PowerRegression::getExponent() {
    return {this->A, std::sqrt(this->sigma2_A)};
}

Quantity PowerRegression::getMultiplier() {
    float multiplier = std::exp(this->lnB);
    return {multiplier, std::sqrt(this->sigma2_lnB) * multiplier};
}

int PowerRegression::size() {
    return this->data.size();
}
