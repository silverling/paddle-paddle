// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/seed_op.h"

#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/attribute_checker.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
class CPUContext;
}  // namespace phi

namespace paddle {
namespace framework {
class OpDesc;
template <typename T> class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative

namespace operators {

class SeedOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim("Out", {1});
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::INT32, ctx.GetPlace());
  }
};

class SeedOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "The output of seed op.");
    AddAttr<int>("seed", "Dropout random seed.").SetDefault(0);
    AddComment(R"DOC(
Seed Operator.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    seed,
    ops::SeedOp,
    ops::SeedOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
PD_REGISTER_STRUCT_KERNEL(seed, CPU, ALL_LAYOUT, ops::CPUSeedKernel, int) {}

/* ==========================  register checkpoint ===========================*/
REGISTER_OP_VERSION(seed).AddCheckpoint(
    R"ROC(
             Upgrade seed add a new attribute [force_cpu])ROC",
    paddle::framework::compatible::OpVersionDesc().NewAttr(
        "force_cpu",
        "If true, Force fill output variable to cpu."
        "memory. Otherwise, fill output variable to the running "
        "device",
        false));
