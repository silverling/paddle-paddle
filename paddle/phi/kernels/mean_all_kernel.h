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

#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {
class DenseTensor;

// In order to be compatible with `mean` op in fluid,
// it is no longer used in 2.x API. It can not implement by call
// ReduceMeanKernel because ReduceMeanKernel doesn't support bfloat16 now,
// maybe we can unify this kernel to ReduceMeanKernel series in the future
template <typename T, typename Context>
void MeanAllKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* out);

}  // namespace phi
