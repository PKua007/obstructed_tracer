/*
 * RandomWalkerFactory.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef RANDOMWALKERFACTORY_H_
#define RANDOMWALKERFACTORY_H_

#include "RandomWalker.h"

class RandomWalkerFactory {
public:
    virtual ~RandomWalkerFactory() = default;

    virtual RandomWalker &getRandomWalker() = 0;
};

#endif /* RANDOMWALKERFACTORY_H_ */
