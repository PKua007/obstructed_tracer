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

void PowerRegression::addXY(double x, double y) {
    Expects(x > 0);
    Expects(y > 0);
    this->data.push_back({x, y});
}

void PowerRegression::calculate(int from, int to) {
    int n = to - from;
    Expects(n > 2);

    double sum_lnx{};
    double sum_lny{};
    double sum_lnxlny{};
    double sum_lnx2{};
    double sum_lny2{};
    for (int i = from; i < to; i++) {
        DataElement d = this->data[i];
        double lnx = std::log(d.x);
        double lny = std::log(d.y);

        sum_lnxlny += lnx*lny;
        sum_lnx += lnx;
        sum_lny += lny;
        sum_lnx2 += lnx*lnx;
        sum_lny2 += lny*lny;
    }
    this->A = (n*sum_lnxlny - sum_lnx*sum_lny) / (n*sum_lnx2 - sum_lnx*sum_lnx);
    this->lnB = (sum_lny - this->A*sum_lnx)/n;

    // SSE, SST and MSE symbols and formulas as in https://www.cse.wustl.edu/~jain/iucee/ftp/k_14slr.pdf
    double sse{};
    for (int i = from; i < to; i++) {
        DataElement d = this->data[i];
        sse += std::pow(std::log(d.y) - this->lnB - this->A*std::log(d.x), 2);
    }
    double sst = sum_lny2 - sum_lny*sum_lny/n;
    Assert(sst >= 0);
    double mse = sse/(n - 2);

    this->sigma2_A = (n/(n*sum_lnx2 - sum_lnx*sum_lnx))*mse;
    this->sigma2_lnB = sum_lnx2/(n*sum_lnx2 - sum_lnx*sum_lnx)*mse;
    this->R2 = (sst - sse)/sst;
}

void PowerRegression::calculate() {
    this->calculate(0, this->data.size());
}

Quantity PowerRegression::getExponent() const {
    return {this->A, std::sqrt(this->sigma2_A)};
}

Quantity PowerRegression::getMultiplier() const {
    double multiplier = std::exp(this->lnB);
    return {multiplier, std::sqrt(this->sigma2_lnB) * multiplier};
}

int PowerRegression::size() const {
    return this->data.size();
}

double PowerRegression::getR2() const {
    return this->R2;
}
