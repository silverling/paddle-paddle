// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <memory>
#include <string>

#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/enforce.h"
#include "gtest/gtest_pred_impl.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace framework {
namespace ir {

template <typename T = float>
void AddVarToScope(Scope* param_scope,
                   const std::string& name,
                   const DDim& dims,
                   T value = 0) {
  auto* tensor = param_scope->Var(name)->GetMutable<phi::DenseTensor>();
  tensor->Resize(dims);
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      platform::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  auto* data = cpu_ctx->Alloc<T>(tensor);
  for (int64_t i = 0; i < tensor->numel(); i++) {
    data[i] = value;
  }
}

TEST(Relu6FusePass, basic) {
  Layers layers;

  auto* in_x = layers.data("in_x", {1, 32, 112, 112});
  auto* clip_min = layers.data("clip_x", {1}, true);
  auto* clip_max = layers.data("clip_y", {1}, true);
  layers.clip(in_x, clip_min, clip_max);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto* param_scope = new Scope();
  graph->Set("__param_scope__", param_scope);
  AddVarToScope(param_scope, clip_min->Name(), {1}, 0.f);
  AddVarToScope(param_scope, clip_max->Name(), {1}, 6.f);
  auto pass = PassRegistry::Instance().Get("relu6_fuse_pass");
  VLOG(3) << DebugString(graph);

  pass->Apply(graph.get());
  VLOG(3) << DebugString(graph);

  auto clip_num = GetNumOpNodes(graph, "clip");
  PADDLE_ENFORCE_EQ(clip_num,
                    0,
                    platform::errors::PreconditionNotMet(
                        "clip should be mapped to relu6 after pass."));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(relu6_fuse_pass);
