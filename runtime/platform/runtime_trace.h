/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Deliberately independent of ET_LOG levels, including in Release builds.
#if defined(ET_RUNTIME_TRACE) && ET_RUNTIME_TRACE
#include <cstdio>
#define ET_TRACE(...)                       \
  do {                                      \
    std::printf("[ET_TRACE] " __VA_ARGS__); \
  } while (0)
#else
#define ET_TRACE(...) \
  do {                \
  } while (0)
#endif
