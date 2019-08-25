/*
 * Move.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef MOVE_H_
#define MOVE_H_

struct Move {
    float x;
    float y;
};

inline Move operator+(Move m1, Move m2) {
    return {m1.x + m2.x, m1.y + m2.y};
}

#endif /* MOVE_H_ */
