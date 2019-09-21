/*
 * Quantity.h
 *
 *  Created on: 22 wrz 2019
 *      Author: pkua
 */

#ifndef QUANTITY_H_
#define QUANTITY_H_

#include <ostream>

struct Quantity {
    float value{};
    float error{};
};

inline std::ostream &operator<<(std::ostream &out, Quantity quantity) {
    out << quantity.value << " ± " << quantity.error;
    return out;
}

#endif /* QUANTITY_H_ */
