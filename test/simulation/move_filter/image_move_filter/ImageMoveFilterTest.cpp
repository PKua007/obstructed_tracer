/*
 * ImageMoveFilterTest.cpp
 *
 *  Created on: 3 sty 2020
 *      Author: pkua
 */

#include <catch2/catch.hpp>
#include <memory>
#include <fstream>

#include "simulation/move_filter/image_move_filter/ImageMoveFilter.h"
#include "simulation/move_filter/image_move_filter/WallBoundaryConditions.h"
#include "simulation/move_filter/image_move_filter/PeriodicBoundaryConditions.h"
#include "image/PPMImageReader.h"

namespace {
    Image read_image(std::string filename) {
        std::fstream imageFile("sample_images/" + filename);
        REQUIRE(imageFile.is_open());
        PPMImageReader reader;
        return reader.read(imageFile);
    }

    template<typename ImageBoundaryConditions>
    std::unique_ptr<ImageMoveFilter<ImageBoundaryConditions>> get_image_move_filter(std::string filename,
                                                                                    unsigned long seed = 1234)
    {
        Image image = read_image(filename);
        return std::unique_ptr<ImageMoveFilter<ImageBoundaryConditions>>(
            new ImageMoveFilter<ImageBoundaryConditions>(image.getIntData().data(), image.getWidth(), image.getHeight(),
                                                         seed, 1)
        );
    }
}

TEST_CASE("ImageMoveFilter: vertical and horizontal lines") {
    auto filter = get_image_move_filter<WallBoundaryConditions>("test_image.ppm");
    filter->setupForTracerRadius(0);

    SECTION("no obstructions") {
        REQUIRE(filter->isMoveValid(Tracer{Point{12, 24}, 0}, Move{10, 0}));
        REQUIRE(filter->isMoveValid(Tracer{Point{12, 24}, 0}, Move{0, -5}));
    }

    SECTION("from wall to wall") {
        SECTION("horizontal") {
            REQUIRE(filter->isMoveValid(Tracer{Point{1, 19}, 0}, Move{47, 0}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{0, 19}, 0}, Move{48, 0}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{1, 19}, 0}, Move{48, 0}));
        }

        SECTION("vertical") {
            REQUIRE(filter->isMoveValid(Tracer{Point{32, 1}, 0}, Move{0, 47}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{32, 0}, 0}, Move{0, 48}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{31, 1}, 0}, Move{0, 48}));
        }
    }

    SECTION("narrow tunnel") {
        SECTION("not-obstructed") {
            REQUIRE(filter->isMoveValid(Tracer{Point{41, 46}, 0}, Move{0, -4}));
        }

        SECTION("obstructed") {
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{41, 46}, 0}, Move{0, -11}));
        }
    }
}

TEST_CASE("ImageMoveFilter: skew lines") {
    auto filter = get_image_move_filter<WallBoundaryConditions>("test_image.ppm");
    filter->setupForTracerRadius(0);

    SECTION("no obstructions") {
        REQUIRE(filter->isMoveValid(Tracer{Point{12, 24}, 0}, Move{10, 2}));
        REQUIRE(filter->isMoveValid(Tracer{Point{12, 24}, 0}, Move{3, -5}));
    }

    SECTION("from wall to wall") {
        REQUIRE(filter->isMoveValid(Tracer{Point{1, 18}, 0}, Move{47, 3}));
        REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{0, 18}, 0}, Move{48, 3}));
        REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{1, 18}, 0}, Move{48, 3}));
        REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{1, 17}, 0}, Move{47, 4}));
        REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{1, 18}, 0}, Move{47, 4}));
    }

    SECTION("narrow tunnel") {
        SECTION("not-obstructed") {
            REQUIRE(filter->isMoveValid(Tracer{Point{38, 8}, 0}, Move{3, 3}));
            REQUIRE(filter->isMoveValid(Tracer{Point{18, 43}, 0}, Move{-5, -2}));
        }

        SECTION("obstructed") {
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{38, 8}, 0}, Move{7, 7}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{18, 43}, 0}, Move{-13, -6}));
        }
    }
}

TEST_CASE("ImageMoveFilter: thick lines") {
    SECTION("radius 1") {
        auto filter = get_image_move_filter<WallBoundaryConditions>("test_image.ppm");
        filter->setupForTracerRadius(1);

        REQUIRE(filter->isMoveValid(Tracer{Point{17, 19}, 1}, Move{0, -6}));
        REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{11, 19}, 1}, Move{0, -6}));
    }

    SECTION("radius 2") {
        auto filter = get_image_move_filter<WallBoundaryConditions>("test_image.ppm");
        filter->setupForTracerRadius(2);

        REQUIRE(filter->isMoveValid(Tracer{Point{7, 12}, 2}, Move{0, -8}));
        REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{17, 20}, 2}, Move{0, -8}));
    }
}

TEST_CASE("ImageMoveFilter: boundary conditions") {
    SECTION("wall") {
        auto filter = get_image_move_filter<WallBoundaryConditions>("test_image.ppm");
        filter->setupForTracerRadius(0);

        SECTION("left") {
            REQUIRE(filter->isMoveValid(Tracer{Point{3, 25}, 0}, Move{-3, 0}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{3, 25}, 0}, Move{-4, 0}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{3, 25}, 0}, Move{-10, 0}));
        }

        SECTION("right") {
            REQUIRE(filter->isMoveValid(Tracer{Point{46, 25}, 0}, Move{3, 0}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{46, 25}, 0}, Move{4, 0}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{46, 25}, 0}, Move{10, 0}));
        }

        SECTION("up") {
            REQUIRE(filter->isMoveValid(Tracer{Point{25, 46}, 0}, Move{0, 3}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{25, 46}, 0}, Move{0, 4}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{25, 46}, 0}, Move{0, 10}));
        }

        SECTION("down") {
            REQUIRE(filter->isMoveValid(Tracer{Point{25, 3}, 0}, Move{0, -3}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{25, 3}, 0}, Move{0, -4}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{25, 3}, 0}, Move{0, -10}));
        }

        SECTION("down-left") {
            REQUIRE(filter->isMoveValid(Tracer{Point{2, 2}, 0}, Move{-2, -2}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{2, 2}, 0}, Move{-3, -3}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{2, 2}, 0}, Move{-6, -6}));
        }

        SECTION("down-right") {
            REQUIRE(filter->isMoveValid(Tracer{Point{47, 2}, 0}, Move{2, -2}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{47, 2}, 0}, Move{3, -3}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{47, 2}, 0}, Move{6, -6}));
        }

        SECTION("up-left") {
            REQUIRE(filter->isMoveValid(Tracer{Point{2, 47}, 0}, Move{-2, 2}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{2, 47}, 0}, Move{-3, 3}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{2, 47}, 0}, Move{-6, 6}));
        }

        SECTION("up-right") {
            REQUIRE(filter->isMoveValid(Tracer{Point{47, 47}, 0}, Move{2, 2}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{47, 47}, 0}, Move{3, 3}));
            REQUIRE_FALSE(filter->isMoveValid(Tracer{Point{47, 47}, 0}, Move{6, 6}));
        }
    }

    SECTION("periodic") {
        auto filter = get_image_move_filter<PeriodicBoundaryConditions>("test_image.ppm");
        filter->setupForTracerRadius(0);

        SECTION("left") {
            REQUIRE(filter->isMoveValid(Tracer{Point{3, 25}, 0}, Move{-3, 0}));
            REQUIRE(filter->isMoveValid(Tracer{Point{3, 25}, 0}, Move{-10, 0}));
        }

        SECTION("right") {
            REQUIRE(filter->isMoveValid(Tracer{Point{46, 25}, 0}, Move{3, 0}));
            REQUIRE(filter->isMoveValid(Tracer{Point{46, 25}, 0}, Move{10, 0}));
        }

        SECTION("up") {
            REQUIRE(filter->isMoveValid(Tracer{Point{25, 46}, 0}, Move{0, 3}));
            REQUIRE(filter->isMoveValid(Tracer{Point{25, 46}, 0}, Move{0, 10}));
        }

        SECTION("down") {
            REQUIRE(filter->isMoveValid(Tracer{Point{25, 3}, 0}, Move{0, -3}));
            REQUIRE(filter->isMoveValid(Tracer{Point{25, 3}, 0}, Move{0, -10}));
        }

        SECTION("down-left") {
            REQUIRE(filter->isMoveValid(Tracer{Point{2, 2}, 0}, Move{-2, -2}));
            REQUIRE(filter->isMoveValid(Tracer{Point{2, 2}, 0}, Move{-6, -6}));
        }

        SECTION("down-right") {
            REQUIRE(filter->isMoveValid(Tracer{Point{47, 2}, 0}, Move{2, -2}));
            REQUIRE(filter->isMoveValid(Tracer{Point{47, 2}, 0}, Move{6, -6}));
        }

        SECTION("up-left") {
            REQUIRE(filter->isMoveValid(Tracer{Point{2, 47}, 0}, Move{-2, 2}));
            REQUIRE(filter->isMoveValid(Tracer{Point{2, 47}, 0}, Move{-6, 6}));
        }

        SECTION("up-right") {
            REQUIRE(filter->isMoveValid(Tracer{Point{47, 47}, 0}, Move{2, 2}));
            REQUIRE(filter->isMoveValid(Tracer{Point{47, 47}, 0}, Move{6, 6}));
        }
    }

    SECTION("periodic far jumps") {
        auto filter = get_image_move_filter<PeriodicBoundaryConditions>("test_image.ppm");
        filter->setupForTracerRadius(0);

        SECTION("left") {
            REQUIRE(filter->isMoveValid(Tracer{Point{3, 25}, 0}, Move{-1000, 0}));
        }

        SECTION("right") {
            REQUIRE(filter->isMoveValid(Tracer{Point{46, 25}, 0}, Move{1000, 0}));
        }

        SECTION("up") {
            REQUIRE(filter->isMoveValid(Tracer{Point{25, 46}, 0}, Move{0, 1000}));
        }

        SECTION("down") {
            REQUIRE(filter->isMoveValid(Tracer{Point{25, 3}, 0}, Move{0, -1000}));
        }
    }
}

TEST_CASE("ImageMoveFilter: available tracers") {
    auto filter = get_image_move_filter<WallBoundaryConditions>("test_image2.ppm");

    SECTION("radius 0") {
        filter->setupForTracerRadius(0);

        REQUIRE(filter->getNumberOfValidTracers() == 10);
        Tracer validTracer = filter->randomValidTracer();
        REQUIRE(validTracer.getRadius() == 0);
    }

    SECTION("radius 1") {
        filter->setupForTracerRadius(1);

        REQUIRE(filter->getNumberOfValidTracers() == 1);
        Tracer validTracer = filter->randomValidTracer();
        REQUIRE(validTracer.getRadius() == 1);
        REQUIRE(validTracer.getPosition().x >= 14);
        REQUIRE(validTracer.getPosition().y >= 10);
        REQUIRE(validTracer.getPosition().x < 15);
        REQUIRE(validTracer.getPosition().y < 11);
    }
}
