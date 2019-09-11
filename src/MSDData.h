/*
 * MSDData.h
 *
 *  Created on: 11 wrz 2019
 *      Author: pkua
 */

#ifndef MSDDATA_H_
#define MSDDATA_H_

#include <iosfwd>

class MSDData {
public:
    virtual ~MSDData() = default;

    virtual void store(std::ostream &out) = 0;
};

#endif /* MSDDATA_H_ */
