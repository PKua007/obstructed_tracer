/*
 * LinearRegression.cpp
 *
 *  Created on: 22.06.2017
 *      Author: ciesla, pkua
 */

#include "PowerRegression.h"

#include <algorithm>
#include <cmath>

void PowerRegression::clear() {
	this->data.clear();
}

/**
 * Add data point
 * @param x
 * @param y
 */
void PowerRegression::addXY(float x, float y) {
	this->data.push_back({x, y});
}

/**
 * calculates fit values
 */
void PowerRegression::calculate(int from, int to) {
    float slnx{};
    float slny{};
    float slnxlny{};
    float slnx2{};
	for (int i = from; i < to; i++) {
		DataElement d = this->data[i];
		float lnx = std::log(d.x);
		float lny = std::log(d.y);
		slnxlny += lnx*lny;
		slnx += lnx;
		slny += lny;
		slnx2 += lnx*lnx;
	}
	this->a = ((to - from)*slnxlny - slnx*slny) / ((to - from)*slnx2 - slnx*slnx);
	this->b = (slny - this->a*slnx)/(to-from);

	float s2{};
	for (int i = from; i < to; i++) {
		DataElement d = this->data[i];
		s2 += std::pow(std::log(d.y) - this->b - this->a*std::log(d.x), 2.);
	}
	this->s2a = ((to - from)/((to - from)*slnx2 - slnx*slnx))*((s2)/(to - from - 2.));
}

/**
 * calculates fit values
 */
void PowerRegression::calculate() {
	this->calculate(0, this->data.size());
}

/**
 * @return parameter A from y = Bx^A
 */
float PowerRegression::getA() {
	return this->a;
}

/**
 * @return standard deviation squared for parameter A from y = Ax + B
 */
float PowerRegression::getSA() {
	return std::sqrt(this->s2a);
}

/**
 * @return parameter B from y = Bx^A
 */
float PowerRegression::getB() {
	return std::exp(this->b);
}

int PowerRegression::size() {
	return this->data.size();
}
