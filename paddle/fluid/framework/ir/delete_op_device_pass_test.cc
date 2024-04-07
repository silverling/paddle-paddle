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

#include <memory>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/ir/pass.h"
#include "gtest/gtest_pred_impl.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(delete_op_device_pass, relu) {
  ProgramDesc program;
  auto* x_var = program.MutableBlock(0)->Var("relu_x");
  auto* out_var = program.MutableBlock(0)->Var("relu_out");
  OpDesc* relu_op = program.MutableBlock(0)->AppendOp();
  relu_op->SetType("relu");
  relu_op->SetInput("X", {x_var->Name()});
  relu_op->SetOutput("Out", {out_var->Name()});
  relu_op->SetAttr("op_device", std::string{"gpu:0"});

  std::unique_ptr<Graph> graph(new Graph(program));
  auto pass = PassRegistry::Instance().Get("delete_op_device_pass");
  graph.reset(pass->Apply(graph.release()));
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) continue;
    if (node->Op()->Type() == "relu") {
      PADDLE_ENFORCE(!node->Op()->HasAttr("op_device"),
                     platform::errors::InvalidArgument(
                         "Run delete_op_device_pass failed. Relu op still has "
                         "'op_device' attr."));
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(delete_op_device_pass);
