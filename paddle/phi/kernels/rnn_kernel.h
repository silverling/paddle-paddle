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

#include <string>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/utils/optional.h"

namespace phi {
class DenseTensor;

template <typename T, typename Context>
void RnnKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<const DenseTensor*>& pre_state,
               const std::vector<const DenseTensor*>& weight_list,
               const paddle::optional<DenseTensor>& sequence_length,
               float dropout_prob,
               bool is_bidirec,
               int input_size,
               int hidden_size,
               int num_layers,
               const std::string& mode,
               int seed,
               bool is_test,
               DenseTensor* out,
               DenseTensor* dropout_state,
               std::vector<DenseTensor*> state,
               DenseTensor* reserve);

}  // namespace phi
