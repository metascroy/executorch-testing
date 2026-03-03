#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -eu

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate et-testing

export CMAKE_OUT=cmake-out
export CMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')

cmake -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} \
    -DCMAKE_INSTALL_PREFIX=${CMAKE_OUT} \
    -Dprotobuf_BUILD_TESTS=OFF \
    -S . \
    -B ${CMAKE_OUT}
cmake --build ${CMAKE_OUT} -j 16 --config Release

echo "Successfully built tests."
