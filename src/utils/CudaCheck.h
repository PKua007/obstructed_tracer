/*
 * CudaCheck.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef CUDACHECK_H_
#define CUDACHECK_H_

#include <stdexcept>

/**
 * @brief An exception thrown by cudaCheck if an error has been encountered.
 */
class CudaException : public std::runtime_error {
public:
    CudaException(const std::string &what) : std::runtime_error{what} { }
};

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const std::string &file, int line) {
    if (code != cudaSuccess)
        throw CudaException(file + ":" + std::to_string(line) + " " + cudaGetErrorString(code));
}

#endif /* CUDACHECK_H_ */
