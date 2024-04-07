// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "test/cpp/eager/test_utils.h"
#include "gtest/gtest_pred_impl.h"
#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/ddim.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);

namespace egr {

TEST(TensorUtils, Test) {
  // Prepare Device Contexts
  eager_test::InitEnv(paddle::platform::CPUPlace());

  // Prepare Inputs
  std::vector<paddle::Tensor> target_tensors;
  paddle::framework::DDim ddim = common::make_ddim({4, 16, 16, 32});

  // Create Target Tensor
  paddle::Tensor t =
      eager_test::CreateTensorWithValue(ddim,
                                        paddle::platform::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        5.0 /*value*/,
                                        true /*is_leaf*/);

  paddle::Tensor t_grad =
      eager_test::CreateTensorWithValue(ddim,
                                        paddle::platform::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        false /*is_leaf*/);

  CHECK_EQ(EagerUtils::IsLeafTensor(t), true);

  // Test Utils
  eager_test::CompareTensorWithValue<float>(t, 5.0);

  egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(&t);
  *meta->MutableGrad() = t_grad;

  eager_test::CompareGradTensorWithValue<float>(t, 1.0);
}

}  // namespace egr
