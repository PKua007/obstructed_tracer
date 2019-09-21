/*
 * PowerRegression.h
 *
 *  Created on: 22.06.2017
 *      Author: ciesla, pkua
 */

#ifndef POWERREGRESSION_H_
#define POWERREGRESSION_H_

#include <vector>

class PowerRegression {
	struct DataElement {
		float x{};
		float y{};

		DataElement() { }
		DataElement(float x, float y) : x{x}, y{y} { }
	};

private:
	std::vector<DataElement> data;

	float a{};
	float b{};
	float s2a{};

public:
	void clear();
	void addXY(float x, float y);
	void calculate(int from, int to);
	void calculate();
	float getA();
	float getSA();
	float getB();
	int size();
};

#endif /* ANALIZATOR_LINEARREGRESSION_H_ */
