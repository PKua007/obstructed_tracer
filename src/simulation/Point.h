/*
 * Point.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

/** @file */

#ifndef POINT_H_
#define POINT_H_

#include <ostream>

#include "Move.h"
#include "utils/CudaDefines.h"

/**
 * @brief A `__host__ __device__` struct representing a floating precision point.
 * @see operator<<(std::ostream &, Point point)
 */
struct Point  {
    float x{};
    float y{};

    CUDA_HOSTDEV Point() { };
    CUDA_HOSTDEV Point(float x, float y) : x{x}, y{y} { }

    /**
     * @brief Returns point moved by the vector @a move.
     * @param move move to be performed on the point
     * @return moved point
     */
    CUDA_HOSTDEV Point operator+(Move move) const { return {this->x + move.x, this->y + move.y}; }

    /**
     * @brief Returns point moved by the vector opposite @a move.
     * @param move move to be subtracted from the point
     * @return moved point
     */
    CUDA_HOSTDEV Point operator-(Move move) const { return {this->x - move.x, this->y - move.y}; }

    /**
     * @brief Moves the point by the vector @a move.
     * @param move move to be performed on the point
     * @return the reference to this
     */
    CUDA_HOSTDEV Point& operator+=(Move move) {
        this->x += move.x;
        this->y += move.y;
        return *this;
    }

    /**
     * @brief Moves the point by the vector opposite to @a move.
     * @param move move to be subtracted from the point
     * @return the reference to this
     */
    CUDA_HOSTDEV Point& operator-=(Move move) {
        this->x -= move.x;
        this->y -= move.y;
        return *this;
    }
};

/**
 * @brief Stream insertion operator printing point in form "x y"
 * @param out stream to print the point to
 * @param point point to be printed
 * @return the reference to @a out
 */
inline std::ostream &operator<<(std::ostream &out, Point point) {
    out << point.x << " " << point.y;
    return out;
}

/**
 * @brief Checks if @a p1 and @a p2 are equal.
 */
inline bool operator==(Point p1, Point p2) {
    return p1.x == p2.x && p1.y == p2.y;
}

/**
 * @brief Checks if @a p1 and @a p2 are not equal.
 */
inline bool operator!=(Point p1, Point p2) {
    return !(p1 == p2);
}

#endif /* POINT_H_ */
