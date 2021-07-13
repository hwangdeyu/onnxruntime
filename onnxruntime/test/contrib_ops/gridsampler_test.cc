// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

TEST(GridsamplerContribOpTest, gridsampler_default) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 4, 4}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  test.AddInput<float>("Grid", {1, 6, 6, 2},
    {-1.0000, -1.0000, -0.6000, -1.0000, -0.2000, -1.0000,  0.2000, -1.0000,
     0.6000, -1.0000,  1.0000, -1.0000, -1.0000, -0.6000, -0.6000, -0.6000,
     -0.2000, -0.6000,  0.2000, -0.6000,  0.6000, -0.6000,  1.0000, -0.6000,
     -1.0000, -0.2000, -0.6000, -0.2000, -0.2000, -0.2000,  0.2000, -0.2000,
     0.6000, -0.2000,  1.0000, -0.2000, -1.0000,  0.2000, -0.6000,  0.2000,
     -0.2000,  0.2000,  0.2000,  0.2000,  0.6000,  0.2000,  1.0000,  0.2000,
     -1.0000,  0.6000, -0.6000,  0.6000, -0.2000,  0.6000,  0.2000,  0.6000,
     0.6000,  0.6000,  1.0000,  0.6000, -1.0000,  1.0000, -0.6000,  1.0000,
     -0.2000,  1.0000,  0.2000,  1.0000,  0.6000,  1.0000,  1.0000,  1.0000
    });
  int64_t align_corners = 0;
  test.AddAttribute("mode", "bilinear");
  test.AddAttribute("padding_mode", "zeros");
  test.AddAttribute("align_corners", align_corners);
  test.AddOutput<float>("Y", {1, 1, 6, 6},
    {0.0000, 0.1500, 0.5500, 0.9500, 1.3500, 0.7500,
     0.6000, 1.5000, 2.3000, 3.1000, 3.9000, 2.1000,
     2.2000, 4.7000, 5.5000, 6.3000, 7.1000, 3.7000,
     3.8000, 7.9000, 8.7000, 9.5000, 10.3000, 5.3000,
     5.4000, 11.1000, 11.9000, 12.7000, 13.5000, 6.9000,
     3.0000, 6.1500, 6.5500, 6.9500, 7.3500, 3.7500
    });
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_paddingmode_zeros) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-10.0000, -10.0000, -5.0000, -5.0000,
     -0.2000, -0.2000, 10.0000, 10.0000,
     10.0000, 10.0000, -0.2000, -0.2000,
     5.0000, 5.0000, 10.0000, 10.0000
    });
  test.AddAttribute("padding_mode", "zeros");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {0.0000, 0.0000, 1.7000, 0.0000, 0.0000, 1.7000, 0.0000, 0.0000});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_paddingmode_border) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-10.0000, -10.0000, -5.0000, -5.0000,
     -0.2000, -0.2000, 10.0000, 10.0000,
     10.0000, 10.0000, -0.2000, -0.2000,
     5.0000, 5.0000, 10.0000, 10.0000
    });
  test.AddAttribute("padding_mode", "border");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {5.0000, 0.0000, 1.7000, 5.0000, 5.0000, 1.7000, 5.0000, 5.0000});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_paddingmode_reflection) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-10.0000, -10.0000, -5.0000, -5.0000,
     -0.2000, -0.2000, 10.0000, 10.0000,
     10.0000, 10.0000, -0.2000, -0.2000,
     5.0000, 5.0000, 10.0000, 10.0000
    });
  test.AddAttribute("padding_mode", "reflection");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {2.5000, 0.0000, 1.7000, 2.5000, 2.5000, 1.7000, 5.0000, 2.5000});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_aligncorners_true) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-1.0000, -1.0000, -0.5000, -0.5000,
     -0.2000, -0.2000, 0.0000, 0.0000,
     0.0000, 0.0000, -0.2000, -0.2000,
     0.5000, 0.5000, 1.0000, 1.0000
    });
  int64_t align_corners = 1;
  test.AddAttribute("mode", "bilinear");
  test.AddAttribute("align_corners", align_corners);
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {0.0000, 1.2500, 2.0000, 2.5000, 2.5000, 2.0000, 3.7500, 5.0000});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_mode_bilinear) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-1.0000, -1.0000, -0.5000, -0.5000,
     -0.2000, -0.2000, 0.0000, 0.0000,
     0.0000, 0.0000, -0.2000, -0.2000,
     0.5000, 0.5000, 1.0000, 1.0000
    });
  test.AddAttribute("mode", "bilinear");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {0.0000, 0.5000, 1.7000, 2.5000, 2.5000, 1.7000, 4.5000, 1.2500});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_mode_nearest) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-1.0000, -1.0000, -0.5000, -0.5000,
     -0.2000, -0.2000, 0.0000, 0.0000,
     0.0000, 0.0000, -0.2000, -0.2000,
     0.5000, 0.5000, 1.0000, 1.0000
    });
  test.AddAttribute("mode", "nearest");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {0., 0., 2., 2., 2., 2., 5., 0.});
  test.Run();
}

TEST(GridsamplerContribOpTest, gridsampler_mode_bicubic) {
  OpTester test("GridSampler", 1, kMSDomain);
  test.AddInput<float>("X", {1, 1, 3, 2}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
  test.AddInput<float>("Grid", {1, 2, 4, 2},
    {-1.0000, -1.0000, -0.5000, -0.5000,
     -0.2000, -0.2000, 0.0000, 0.0000,
     0.0000, 0.0000, -0.2000, -0.2000,
     0.5000, 0.5000, 1.0000, 1.0000
    });
  test.AddAttribute("mode", "bicubic");
  test.AddOutput<float>("Y", {1, 1, 2, 4}, {-0.1406, 0.3828, 1.7556, 2.9688, 2.9688, 1.7556, 5.1445, 1.3906});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
