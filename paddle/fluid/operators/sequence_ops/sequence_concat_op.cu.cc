// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdint.h>
#include <vector>

#include "paddle/fluid/operators/sequence_ops/sequence_concat_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/dense_tensor.inl"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
class GPUContext;
}  // namespace phi

PD_REGISTER_STRUCT_KERNEL(sequence_concat,
                          GPU,
                          ALL_LAYOUT,
                          paddle::operators::SeqConcatKernel,
                          float,
                          double,
                          int,
                          int64_t) {}
PD_REGISTER_STRUCT_KERNEL(sequence_concat_grad,
                          GPU,
                          ALL_LAYOUT,
                          paddle::operators::SeqConcatGradKernel,
                          float,
                          double,
                          int,
                          int64_t) {}
