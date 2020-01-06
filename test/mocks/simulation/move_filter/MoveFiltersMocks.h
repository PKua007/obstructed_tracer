/*
 * MoveFiltersMocks.h
 *
 *  Created on: 25 gru 2019
 *      Author: pkua
 */

#ifndef MOVEFILTERSMOCKS_H_
#define MOVEFILTERSMOCKS_H_

#include "simulation/MoveFilter.h"
#include "utils/CudaDefines.h"

#include "test_utils/GPURTTI.h"

struct DefaultMoveFilterMock : public MoveFilter, GPUNamedClass {
    CUDA_HOSTDEV bool isMoveValid(Tracer tracer, Move move) const override { return false; }
    CUDA_HOSTDEV void setupForTracerRadius(float radius) override { }
    CUDA_HOSTDEV Tracer randomValidTracer() override { return Tracer{Point{0, 0}, 0}; }
    CUDA_IMPLEMENT_GET_CLASS_NAME("DefaultMoveFilterMock");
};

struct ImageMoveFilterMock : public MoveFilter, GPUNamedClass {
    size_t width;
    size_t height;
    unsigned int *intImageData;
    unsigned long seed;
    size_t numberOfTrajectories;

    CUDA_HOSTDEV ImageMoveFilterMock(unsigned int *intImageData, size_t width, size_t height, unsigned long seed,
                                     size_t numberOfTrajectories)
            : width{width}, height{height}, seed{seed}, numberOfTrajectories{numberOfTrajectories}
    {
        this->intImageData = new unsigned int[width*height];
        for (size_t i = 0; i < width*height; i++)
            this->intImageData[i] = intImageData[i];
    }

    ImageMoveFilterMock(const ImageMoveFilterMock &other) = delete;
    ImageMoveFilterMock &operator=(const ImageMoveFilterMock &other) = delete;

    CUDA_HOSTDEV ~ImageMoveFilterMock() {
        delete this->intImageData;
    }

    CUDA_HOSTDEV bool isMoveValid(Tracer tracer, Move move) const override { return false; }
    CUDA_HOSTDEV void setupForTracerRadius(float radius) override { }
    CUDA_HOSTDEV Tracer randomValidTracer() override { return Tracer{Point{0, 0}, 0}; }
};

struct ImageMoveFilterWallBCMock : public ImageMoveFilterMock {
    CUDA_HOSTDEV ImageMoveFilterWallBCMock(unsigned int *intImageData, size_t width, size_t height, unsigned long seed,
                                           size_t numberOfTrajectories)
            : ImageMoveFilterMock(intImageData, width, height, seed, numberOfTrajectories)
    { }

    CUDA_IMPLEMENT_GET_CLASS_NAME("ImageMoveFilterWallBCMock");
};

struct ImageMoveFilterPeriodicBCMock : public ImageMoveFilterMock {
    CUDA_HOSTDEV ImageMoveFilterPeriodicBCMock(unsigned int *intImageData, size_t width, size_t height,
                                               unsigned long seed, size_t numberOfTrajectories)
            : ImageMoveFilterMock(intImageData, width, height, seed, numberOfTrajectories)
    { }

    CUDA_IMPLEMENT_GET_CLASS_NAME("ImageMoveFilterPeriodicBCMock");
};

#endif /* MOVEFILTERSMOCKS_H_ */
