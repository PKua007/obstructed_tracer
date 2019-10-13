/*
 * Quantity.h
 *
 *  Created on: 22 wrz 2019
 *      Author: pkua
 */

/** @file */

#ifndef QUANTITY_H_
#define QUANTITY_H_

#include <ostream>

/**
 * @brief A struct representing a quantity with error.
 * @see operator<<(std::ostream &, Quantity)
 */
struct Quantity {
    double value{};
    double error{};

    Quantity() { }
    Quantity(double value, double error) : value{value}, error{error} { }
};

/**
 * @brief Stream insertion operator for Quantity printing in 23.56 &plusmn; 0.21 form.
 * @param out stream to print @a quantity to
 * @param quantity Quantity to be printed
 * @return reference to @a out
 */
inline std::ostream &operator<<(std::ostream &out, Quantity quantity) {
    out << quantity.value << " ± " << quantity.error;
    return out;
}

#endif /* QUANTITY_H_ */
