/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/platform/runtime_trace.h>

#if defined(ET_RUNTIME_TRACE) && ET_RUNTIME_TRACE
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <cinttypes>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {
namespace internal {

// Metadata only: never dereference tensor data (outputs may be uninitialized).
inline void trace_tensor(const char* role, const executorch::aten::Tensor& t) {
  ET_TRACE(
      "  %s: data=%p dtype=%s(%d) numel=%zu nbytes=%zu sizes=[",
      role,
      t.const_data_ptr(),
      ::executorch::runtime::toString(t.scalar_type()),
      static_cast<int>(t.scalar_type()),
      static_cast<size_t>(t.numel()),
      static_cast<size_t>(t.nbytes()));
  for (size_t d = 0; d < static_cast<size_t>(t.dim()); ++d) {
    std::printf("%s%" PRId64, d ? "," : "", static_cast<int64_t>(t.size(d)));
  }
  std::printf("] strides=[");
  for (size_t d = 0; d < static_cast<size_t>(t.dim()); ++d) {
    std::printf(
        "%s%" PRId64, d ? "," : "", static_cast<int64_t>(t.strides()[d]));
  }
  std::printf("]\n");
}

inline void trace_evalue(const EValue& v) {
  ET_TRACE(
      "  EValue=%p tag=%u",
      static_cast<const void*>(&v),
      static_cast<unsigned>(v.tag));
  if (v.isTensor()) {
    std::printf("\n");
    trace_tensor("tensor", v.toTensor());
  } else if (v.isInt()) {
    std::printf(" int=%" PRId64 "\n", v.toInt());
  } else if (v.isBool()) {
    std::printf(" bool=%d\n", static_cast<int>(v.toBool()));
  } else if (v.isDouble()) {
    std::printf(" double=%g\n", v.toDouble());
  } else if (v.isIntList()) {
    std::printf(" int_list=[");
    const auto list = v.toIntList();
    for (size_t i = 0; i < list.size(); ++i) {
      std::printf("%s%" PRId64, i ? "," : "", list[i]);
    }
    std::printf("]\n");
  } else if (v.isTensorList()) {
    std::printf(" tensor_list\n");
    for (const auto& t : v.toTensorList()) {
      trace_tensor("list item", t);
    }
  } else {
    std::printf("\n");
  }
}

} // namespace internal
} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
#define ET_TRACE_TENSOR(role, tensor) \
  ::executorch::runtime::internal::trace_tensor(role, tensor)
#define ET_TRACE_EVALUE(value) \
  ::executorch::runtime::internal::trace_evalue(value)
#else
#define ET_TRACE_TENSOR(role, tensor) \
  do {                                \
  } while (0)
#define ET_TRACE_EVALUE(value) \
  do {                         \
  } while (0)
#endif
