/*
 * GPURandomWalkerTest.cu
 *
 *  Created on: 22 gru 2019
 *      Author: pkua
 */

#include <catch2/catch.hpp>
#include <sstream>

#include "utils/CudaCheck.h"
#include "simulation/MoveFilter.h"
#include "simulation/MoveGenerator.h"
#include "simulation/random_walker/GPURandomWalker.h"

#define GPU_THROW(...) printf(__VA_ARGS__); asm("trap;")

namespace {
    class MoveFilterGPUMock : public MoveFilter {
    private:
        std::size_t validTracerTimesInvoked{};

    public:
        CUDA_HOSTDEV bool isMoveValid(Tracer tracer, Move move) const override {
            if (tracer == Tracer{Point{1, 2}, 3} && move == Move{5, 8}) {
                return false;
            } else if (tracer == Tracer{Point{1, 2}, 3} && move == Move{7, 10}) {
                return true;
            } else {
                GPU_THROW("MoveFilter::isMoveValid: got Tracer{Point{%f, %f}, %f}, Move{%f, %f}",
                          tracer.getPosition().x, tracer.getPosition().y, tracer.getRadius(), move.x, move.y);
                return false;
            }
        }

        CUDA_HOSTDEV void setupForTracerRadius(float radius) override {
            if (!CUDA_IS_IT_FIRST_THREAD)
                return;

            if (radius != 3) {
                GPU_THROW("MoveFilter::setupForTracerRadius: got radius %f", radius);
            }
        }

        CUDA_HOSTDEV Tracer randomValidTracer() override {
            if (CUDA_THREAD_IDX == 0)
                return Tracer{Point{1, 2}, 3};
            else if (CUDA_THREAD_IDX == 1)
                return Tracer{Point{3, 4}, 3};
            else if (CUDA_THREAD_IDX == 2)
                return Tracer{Point{5, 6}, 3};
            else
                return Tracer{Point{0, 0}, 0};
        }
    };

    class MoveGeneratorGPUMock : public MoveGenerator {
    private:
        std::size_t timesInvoked{};

    public:
        CUDA_HOSTDEV virtual Move generateMove() override {
            if (this->timesInvoked == 0) {
                this->timesInvoked++;
                return Move{3, 4};
            } else if (this->timesInvoked == 1) {
                this->timesInvoked++;
                return Move{5, 6};
            } else {
                GPU_THROW("generating move for the third time");
                return Move{};
            }
        }
    };

    __global__
    void gpu_create_move_generator(MoveGenerator **moveGenerator)
    {
        if (!CUDA_IS_IT_FIRST_THREAD)
            return;
        (*moveGenerator) = new MoveGeneratorGPUMock;
    }

    __global__
    void gpu_create_move_filter( MoveFilter **moveFilter)
    {
        if (!CUDA_IS_IT_FIRST_THREAD)
            return;
        (*moveFilter) = new MoveFilterGPUMock;
    }

    MoveGenerator *create_move_generator() {
        MoveGenerator **moveGeneratorPlaceholder{};
        cudaCheck( cudaMalloc(&moveGeneratorPlaceholder, sizeof(MoveGenerator**)) );
        gpu_create_move_generator<<<1, 32>>>(moveGeneratorPlaceholder);
        cudaCheck( cudaDeviceSynchronize() );

        MoveGenerator *moveGenerator;
        cudaCheck( cudaMemcpy(&moveGenerator, moveGeneratorPlaceholder, sizeof(MoveGenerator*),
                              cudaMemcpyDeviceToHost) );
        cudaCheck( cudaFree(moveGeneratorPlaceholder) );

        return moveGenerator;
    }

    MoveFilter *create_move_filter() {
        MoveFilter **moveFilterPlaceholder{};
        cudaCheck( cudaMalloc(&moveFilterPlaceholder, sizeof(MoveFilter**)) );
        gpu_create_move_filter<<<1, 32>>>(moveFilterPlaceholder);
        cudaCheck( cudaDeviceSynchronize() );

        MoveFilter *moveFilter;
        cudaCheck( cudaMemcpy(&moveFilter, moveFilterPlaceholder, sizeof(MoveFilter*),
                              cudaMemcpyDeviceToHost) );
        cudaCheck( cudaFree(moveFilterPlaceholder) );

        return moveFilter;
    }
}


TEST_CASE("GPURandomWalker: sampling valid tracers") {
    RandomWalker::WalkParameters walkParameters;
    walkParameters.numberOfSteps = 2;
    walkParameters.tracerRadius = 3;
    walkParameters.integrationStep = 2;
    walkParameters.drift = Move{1, 2};
    std::ostringstream logger;
    GPURandomWalker gpuRandomWalker(3, walkParameters, 1, create_move_generator(), create_move_filter(), logger);

    auto initialTracers = gpuRandomWalker.getRandomInitialTracersVector();

    REQUIRE(initialTracers.size() == 3);
    REQUIRE(initialTracers[0] == Tracer{Point{1, 2}, 3});
    REQUIRE(initialTracers[1] == Tracer{Point{3, 4}, 3});
    REQUIRE(initialTracers[2] == Tracer{Point{5, 6}, 3});
}

TEST_CASE("GPURandomWalker: run with everything tested (1 trajectory, 1 step rejected, drift rescaling)") {
    RandomWalker::WalkParameters walkParameters;
    walkParameters.numberOfSteps = 2;
    walkParameters.tracerRadius = 3;
    walkParameters.integrationStep = 2;
    walkParameters.drift = Move{1, 2};
    std::ostringstream logger;
    GPURandomWalker gpuRandomWalker(1, walkParameters, 1, create_move_generator(), create_move_filter(), logger);

    // Expected run:
    // {1, 2} ---move{3, 4}+drift{1, 2}*integrationStep{2}---rejected---> {1, 2}
    // {1, 2} ---move{5, 6}+drift{1, 2}*integrationStep{2}---accepted---> {8, 12}
    gpuRandomWalker.run(logger, std::vector<Tracer>{Tracer{Point{1, 2}, 3}});

    REQUIRE(gpuRandomWalker.getNumberOfTrajectories() == 1);
    auto traj = gpuRandomWalker.getTrajectory(0);
    REQUIRE(traj.getSize() == 3);
    REQUIRE(traj[0] == Point{1, 2});
    REQUIRE(traj[1] == Point{1, 2});
    REQUIRE(traj[2] == Point{8, 12});
    REQUIRE(traj.getNumberOfAcceptedSteps() == 1);
}
