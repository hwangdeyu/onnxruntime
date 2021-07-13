# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Test reference implementation and model for ONNX Runtime conrtib op gridsampler

import onnx
import unittest
import numpy as np
from onnx_contrib_ops_helper import expect


class ONNXReferenceImplementationTest(unittest.TestCase):
    def test_gridsampler():  # type: () -> None
        node = onnx.helper.make_node(
            'GridSampler',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='bilinear',
            padding_mode='zeros',
            align_corners=0,
        )
        # X shape, [N, C, H, W] - [1, 1, 4, 4]
        X = np.array(
            [
                [
                    [
                        [0., 1., 2., 3.],
                        [4., 5., 6., 7.],
                        [8., 9., 10., 11.],
                        [12., 13., 14., 15.]
                    ]
                ]
            ],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 6, 6, 2]
        Grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.6000, -1.0000],
                        [-0.2000, -1.0000],
                        [0.2000, -1.0000],
                        [0.6000, -1.0000],
                        [1.0000, -1.0000]
                    ],
                    [
                        [-1.0000, -0.6000],
                        [-0.6000, -0.6000],
                        [-0.2000, -0.6000],
                        [0.2000, -0.6000],
                        [0.6000, -0.6000],
                        [1.0000, -0.6000]
                    ],
                    [
                        [-1.0000, -0.2000],
                        [-0.6000, -0.2000],
                        [-0.2000, -0.2000],
                        [0.2000, -0.2000],
                        [0.6000, -0.2000],
                        [1.0000, -0.2000]
                    ],
                    [
                        [-1.0000, 0.2000],
                        [-0.6000, 0.2000],
                        [-0.2000, 0.2000],
                        [0.2000, 0.2000],
                        [0.6000, 0.2000],
                        [1.0000, 0.2000]
                    ],
                    [
                        [-1.0000, 0.6000],
                        [-0.6000, 0.6000],
                        [-0.2000, 0.6000],
                        [0.2000, 0.6000],
                        [0.6000, 0.6000],
                        [1.0000, 0.6000]
                    ],
                    [
                        [-1.0000, 1.0000],
                        [-0.6000, 1.0000],
                        [-0.2000, 1.0000],
                        [0.2000, 1.0000],
                        [0.6000, 1.0000],
                        [1.0000, 1.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 6, 6]
        Y = np.array(
            [
                [
                    [
                        [0.0000, 0.1500, 0.5500, 0.9500, 1.3500, 0.7500],
                        [0.6000, 1.5000, 2.3000, 3.1000, 3.9000, 2.1000],
                        [2.2000, 4.7000, 5.5000, 6.3000, 7.1000, 3.7000],
                        [3.8000, 7.9000, 8.7000, 9.5000, 10.3000, 5.3000],
                        [5.4000, 11.1000, 11.9000, 12.7000, 13.5000, 6.9000],
                        [3.0000, 6.1500, 6.5500, 6.9500, 7.3500, 3.7500]
                    ]
                ]
            ],
            dtype=np.float32,
        )
        expect(node, inputs=[X, Grid], outputs=[Y],
               name='test_gridsampler')

    @staticmethod
    def test_gridsampler_zeros_padding():  # type: () -> None
        # X shape, [N, C, H, W] - [1, 1, 3, 2]
        X = np.array(
            [
                [
                    [
                        [0., 1.],
                        [2., 3.],
                        [4., 5.]
                    ]
                ]
            ],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
        Grid = np.array(
            [
                [
                    [
                        [-10.0000, -10.0000],
                        [-5.0000, -5.0000],
                        [-0.2000, -0.2000],
                        [10.0000, 10.0000]
                    ],

                    [
                        [10.0000, 10.0000],
                        [-0.2000, -0.2000],
                        [5.0000, 5.0000],
                        [10.0000, 10.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        # setting padding_mode = 'zeros'
        node = onnx.helper.make_node(
            'GridSampler',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            padding_mode='zeros',
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_zeros = np.array(
            [
                [
                    [
                        [0.0000, 0.0000, 1.7000, 0.0000],
                        [0.0000, 1.7000, 0.0000, 0.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, Grid], outputs=[Y_zeros],
               name='test_gridsampler_zeros_padding')

    @staticmethod
    def test_gridsampler_border_padding():  # type: () -> None
        # X shape, [N, C, H, W] - [1, 1, 3, 2]
        X = np.array(
            [
                [
                    [
                        [0., 1.],
                        [2., 3.],
                        [4., 5.]
                    ]
                ]
            ],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
        Grid = np.array(
            [
                [
                    [
                        [-10.0000, -10.0000],
                        [-5.0000, -5.0000],
                        [-0.2000, -0.2000],
                        [10.0000, 10.0000]
                    ],

                    [
                        [10.0000, 10.0000],
                        [-0.2000, -0.2000],
                        [5.0000, 5.0000],
                        [10.0000, 10.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        # setting padding_mode = 'border'
        node = onnx.helper.make_node(
            'GridSampler',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            padding_mode='border',
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_border = np.array(
            [
                [
                    [
                        [5.0000, 0.0000, 1.7000, 5.0000],
                        [5.0000, 1.7000, 5.0000, 5.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, Grid], outputs=[Y_border],
               name='test_gridsampler_border_padding')

    @staticmethod
    def test_gridsampler_reflection_padding():  # type: () -> None
        # X shape, [N, C, H, W] - [1, 1, 3, 2]
        X = np.array(
            [
                [
                    [
                        [0., 1.],
                        [2., 3.],
                        [4., 5.]
                    ]
                ]
            ],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
        Grid = np.array(
            [
                [
                    [
                        [-10.0000, -10.0000],
                        [-5.0000, -5.0000],
                        [-0.2000, -0.2000],
                        [10.0000, 10.0000]
                    ],

                    [
                        [10.0000, 10.0000],
                        [-0.2000, -0.2000],
                        [5.0000, 5.0000],
                        [10.0000, 10.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        # setting padding_mode = 'reflection'
        node = onnx.helper.make_node(
            'GridSampler',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            padding_mode='reflection',
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_reflection = np.array(
            [
                [
                    [
                        [2.5000, 0.0000, 1.7000, 2.5000],
                        [2.5000, 1.7000, 5.0000, 2.5000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, Grid], outputs=[Y_reflection],
               name='test_gridsampler_reflection_padding')

    @staticmethod
    def test_gridsampler_bilinear():  # type: () -> None
        # X shape, [N, C, H, W] - [1, 1, 3, 2]
        X = np.array(
            [
                [
                    [
                        [0., 1.],
                        [2., 3.],
                        [4., 5.]
                    ]
                ]
            ],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
        Grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.5000, -0.5000],
                        [-0.2000, -0.2000],
                        [0.0000, 0.0000]
                    ],

                    [
                        [0.0000, 0.0000],
                        [-0.2000, -0.2000],
                        [0.5000, 0.5000],
                        [1.0000, 1.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        # setting mode = 'bilinear', default align_corners = 0
        node = onnx.helper.make_node(
            'GridSampler',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='bilinear',
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bilinear = np.array(
            [
                [
                    [
                        [0.0000, 0.5000, 1.7000, 2.5000],
                        [2.5000, 1.7000, 4.5000, 1.2500]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, Grid], outputs=[Y_bilinear],
               name='test_gridsampler_bilinear')

    @staticmethod
    def test_gridsampler_aligncorners_true():  # type: () -> None
        # X shape, [N, C, H, W] - [1, 1, 3, 2]
        X = np.array(
            [
                [
                    [
                        [0., 1.],
                        [2., 3.],
                        [4., 5.]
                    ]
                ]
            ],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
        Grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.5000, -0.5000],
                        [-0.2000, -0.2000],
                        [0.0000, 0.0000]
                    ],

                    [
                        [0.0000, 0.0000],
                        [-0.2000, -0.2000],
                        [0.5000, 0.5000],
                        [1.0000, 1.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        # setting mode = 'bilinear', align_corners = 1
        node = onnx.helper.make_node(
            'GridSampler',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='bilinear',
            align_corners=1,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_align_corners = np.array(
            [
                [
                    [
                        [0.0000, 1.2500, 2.0000, 2.5000],
                        [2.5000, 2.0000, 3.7500, 5.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, Grid], outputs=[Y_align_corners],
               name='test_gridsampler_aligncorners_true')

    @staticmethod
    def test_gridsampler_nearest():  # type: () -> None
        # X shape, [N, C, H, W] - [1, 1, 3, 2]
        X = np.array(
            [
                [
                    [
                        [0., 1.],
                        [2., 3.],
                        [4., 5.]
                    ]
                ]
            ],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
        Grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.5000, -0.5000],
                        [-0.2000, -0.2000],
                        [0.0000, 0.0000]
                    ],

                    [
                        [0.0000, 0.0000],
                        [-0.2000, -0.2000],
                        [0.5000, 0.5000],
                        [1.0000, 1.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        # setting mode = 'nearest'
        node = onnx.helper.make_node(
            'GridSampler',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='nearest',
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_nearest = np.array(
            [
                [
                    [
                        [0., 0., 2., 2.],
                        [2., 2., 5., 0.]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, Grid], outputs=[Y_nearest],
               name='test_gridsampler_nearest')

    @staticmethod
    def test_gridsampler_bicubic():  # type: () -> None
        # X shape, [N, C, H, W] - [1, 1, 3, 2]
        X = np.array(
            [
                [
                    [
                        [0., 1.],
                        [2., 3.],
                        [4., 5.]
                    ]
                ]
            ],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
        Grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.5000, -0.5000],
                        [-0.2000, -0.2000],
                        [0.0000, 0.0000]
                    ],

                    [
                        [0.0000, 0.0000],
                        [-0.2000, -0.2000],
                        [0.5000, 0.5000],
                        [1.0000, 1.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        # setting mode = 'bicubic'
        node = onnx.helper.make_node(
            'GridSampler',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='bicubic',
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bicubic = np.array(
            [
                [
                    [
                        [-0.1406, 0.3828, 1.7556, 2.9688],
                        [2.9688, 1.7556, 5.1445, 1.3906]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, Grid], outputs=[Y_bicubic],
               name='test_gridsampler_bicubic')

if __name__ == '__main__':
    unittest.main(module=__name__, buffer=True)