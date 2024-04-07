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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dense_tensor.inl"

namespace paddle {
namespace jit {
class BaseEngine;

using DenseTensor = phi::DenseTensor;
using Tensor = paddle::Tensor;

class Function {
 public:
  Function() : engine_(nullptr) {}
  explicit Function(BaseEngine* engine);

  std::vector<Tensor> operator()(const std::vector<Tensor>& inputs) const;

  std::vector<DenseTensor> operator()(
      const std::vector<DenseTensor>& inputs) const;

  bool IsValid() const { return engine_ != nullptr; }

  ~Function() = default;

 private:
  BaseEngine* engine_;
};

}  // namespace jit
}  // namespace paddle
