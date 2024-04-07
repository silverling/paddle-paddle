/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_registry.h"

namespace paddle {
namespace framework {
class OpDesc;
template <typename T> class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative

namespace operators {

class GetFloatStatusOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutput("FloatStatusOut"),
                   "Output",
                   "FloatStatusOut",
                   "get_float_status");
    ctx->SetOutputDim("FloatStatusOut", ctx->GetInputDim("FloatStatus"));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
  }
};

class GetFloatStatusMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("FloatStatus",
             "(Tensor) of shape {8} that holds the float status.");
    AddOutput("FloatStatusOut",
              "(Tensor) of shape {8} that holds the get float status.");
    AddComment(R"DOC(
      Get the float status
)DOC");
  }
};

template <typename T, typename DeviceContext>
class GetFloatStatusKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Operator get_float_status is not supported on CPU"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = phi::CPUContext;

REGISTER_OPERATOR(
    get_float_status,
    ops::GetFloatStatusOp,
    ops::GetFloatStatusMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

PD_REGISTER_STRUCT_KERNEL(
    get_float_status, CPU, ALL_LAYOUT, ops::GetFloatStatusKernel, float) {}
