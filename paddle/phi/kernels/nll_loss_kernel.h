
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

#include <stdint.h>
#include <string>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/utils/optional.h"

namespace paddle {
template <class T> class optional;
}  // namespace paddle

namespace phi {
class DenseTensor;

template <typename T, typename Context>
void NllLossRawKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& label,
                      const paddle::optional<DenseTensor>& weight,
                      int64_t ignore_index,
                      const std::string& reduction,
                      DenseTensor* out,
                      DenseTensor* total_weight);

}  // namespace phi
