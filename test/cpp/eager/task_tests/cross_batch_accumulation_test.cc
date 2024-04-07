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

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/generated/eager_generated/backwards/scale_node.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/phi/core/kernel_registry.h"
#include "test/cpp/eager/test_utils.h"
#include "gtest/gtest_pred_impl.h"
#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/eager/api/utils/hook_utils.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/ddim.h"

namespace egr {
class GradNodeBase;
}  // namespace egr

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);

namespace egr {

TEST(CrossBatchAccumulation, SingleScaleNode) {
  eager_test::InitEnv(paddle::platform::CPUPlace());

  std::vector<paddle::Tensor> target_tensors;
  paddle::framework::DDim ddim = common::make_ddim({4, 16, 16, 32});

  paddle::Tensor tensor =
      eager_test::CreateTensorWithValue(ddim,
                                        paddle::platform::CPUPlace(),
                                        phi::DataType::FLOAT32,
                                        phi::DataLayout::NCHW,
                                        1.0 /*value*/,
                                        false /*is_leaf*/);
  target_tensors.emplace_back(std::move(tensor));
  paddle::Tensor& target_tensor = target_tensors[0];

  paddle::Tensor leaf_tensor = paddle::Tensor();

  auto scale_node_ptr = std::make_shared<GradNodeScale>(1, 1);
  scale_node_ptr->SetAttributes_scale(5.0 /*scale*/);

  scale_node_ptr->SetDefaultGradInOutMeta();

  AutogradMeta* auto_grad_meta = EagerUtils::autograd_meta(&target_tensor);
  auto_grad_meta->SetGradNode(
      std::dynamic_pointer_cast<GradNodeBase>(scale_node_ptr));
  auto_grad_meta->SetSingleOutRankWithSlot(0, 0);
  auto_grad_meta->SetStopGradient(false);
  egr_utils_api::RetainGradForTensor(target_tensor);  // result: 1.0

  AutogradMeta* meta = EagerUtils::autograd_meta(&leaf_tensor);
  auto acc_node_ptr = std::make_shared<GradNodeAccumulation>(meta);
  meta->SetStopGradient(false);
  meta->SetSingleOutRankWithSlot(0, 0);
  meta->SetGradNode(acc_node_ptr);
  std::vector<egr::AutogradMeta*> res = {meta};
  scale_node_ptr->SetGradOutMeta(leaf_tensor, 0);

  Backward(target_tensors, {});

  eager_test::CompareGradTensorWithValue<float>(target_tensor, 1.0);
  eager_test::CompareGradTensorWithValue<float>(leaf_tensor, 5.0);

  Backward(target_tensors, {});

  eager_test::CompareGradTensorWithValue<float>(target_tensor, 1.0);
  eager_test::CompareGradTensorWithValue<float>(leaf_tensor, 10.0);
}

}  // namespace egr
