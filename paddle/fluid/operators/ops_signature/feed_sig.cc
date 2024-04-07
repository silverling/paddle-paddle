// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>

#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/phi/core/type_defs.h"

namespace phi {

KernelSignature FeedOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorOutput("Out")) {
    return KernelSignature("feed_dense_tensor", {"X"}, {"col"}, {"Out"});
  } else if (ctx.IsSparseCooTensorOutput("Out")) {
    return KernelSignature("feed_sparse_coo_tensor", {"X"}, {"col"}, {"Out"});
  } else {
    return KernelSignature("feed_strings", {"X"}, {"col"}, {"Out"});
  }
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(feed, feed_dense_tensor);
PD_REGISTER_ARG_MAPPING_FN(feed, phi::FeedOpArgumentMapping);
