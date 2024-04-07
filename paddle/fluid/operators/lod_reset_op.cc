/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/lod_reset_op.h"

#include <string>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/attribute_checker.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/grad_op_desc_maker.h"
#include "paddle/fluid/framework/inplace_op_inference.h"
#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
class CPUContext;
}  // namespace phi

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative

namespace operators {

class LoDResetOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "LoDReset");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "LoDReset");

    if (!ctx->HasInput("Y")) {
      auto level0 = ctx->Attrs().Get<std::vector<int>>("target_lod");
      PADDLE_ENFORCE_GT(
          static_cast<int64_t>(level0.size()),
          0,
          platform::errors::InvalidArgument(
              "If Input(Y) is not provided, the output's LoD should be "
              "specified by attribute 'target_lod'. But the size of "
              "'target_lod' is 0."));
    } else if (ctx->IsRuntime()) {
      ctx->ShareLoD("Y", "Out");
    }
    auto append = ctx->Attrs().Get<bool>("append");
    if (append) {
      ctx->ShareLoD("X", /*->*/ "Out");
    }

    if (ctx->HasInput("Y")) {
      if (!ctx->IsRuntime()) {
        ctx->SetLoDLevel("Out", std::max(ctx->GetLoDLevel("Y"), 1));
      }
    } else if (append) {
      if (!ctx->IsRuntime()) {
        ctx->SetLoDLevel("Out", std::max(ctx->GetLoDLevel("X") + 1, 1));
      }
    } else {
      if (!ctx->IsRuntime()) {
        ctx->SetLoDLevel("Out", 1);
      }
    }
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.device_context().GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    return phi::KernelKey(phi::Backend::ALL_BACKEND,
                          tensor.layout(),
                          expected_kernel_type.dtype());
  }
};

class LoDResetOpVarTypeInference
    : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_var_name = Input(ctx, "X").front();
    auto out_var_name = Output(ctx, "Out").front();
    bool append = PADDLE_GET_CONST(bool, ctx->GetAttr("append"));
    if (ctx->HasInput("Y")) {
      auto y_var_name = Input(ctx, "Y").front();
      auto y_lod_level = std::max(GetLoDLevel(ctx, y_var_name), 1);
      SetLoDLevel(ctx, out_var_name, y_lod_level);
    } else if (append) {
      auto x_lod_level = std::max(GetLoDLevel(ctx, x_var_name), 1);
      SetLoDLevel(ctx, out_var_name, x_lod_level);
    } else {
      SetLoDLevel(ctx, out_var_name, 1);
    }
    SetDataType(ctx, out_var_name, GetDataType(ctx, x_var_name));
    SetType(ctx, out_var_name, paddle::framework::proto::VarType::LOD_TENSOR);
  }
};

class LoDResetOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, phi::DenseTensor) Input variable of LoDResetOp which "
             "could be a Tensor or phi::DenseTensor, where the data of output "
             "variable inherits from.");
    AddInput("Y",
             "(phi::DenseTensor, optional) If provided and Y is "
             "phi::DenseTensor, "
             "lod of Input(Y) would be considered as the target lod first, "
             "otherwise data of Input(Y) would be considered as the "
             "target lod.")
        .AsDispensable();
    AddOutput(
        "Out",
        "(phi::DenseTensor) Output variable of LoDResetOp which should be a "
        "phi::DenseTensor.");
    AddAttr<std::vector<int>>("target_lod",
                              "The target level 0 LoD from Attr().")
        .SetDefault(std::vector<int>{});
    AddAttr<bool>("append", "Append data to lod vector.").SetDefault(false);
    AddComment(R"DOC(LoDReset operator

Set LoD of `X` to a new one specified by `Y` or attribute `target_lod`. When `Y`
provided and `Y` is a phi::DenseTensor, `Y.lod` would be considered as target LoD
first, otherwise `Y.data` would be considered as target LoD. If `Y` is not
provided, target LoD should be specified by attribute `target_lod`.
If target LoD is specified by `Y.data` or `target_lod`, only one level LoD
is supported.

Example 1:

Given a 1-level phi::DenseTensor input(X):
    X.lod =  [[ 0,     2,                   5      6 ]]
    X.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    X.dims = [6, 1]

attr(target_lod): [0, 4, 6]

then we get a 1-level LoDTensor:
    Out.lod =  [[ 0,                   4,            6 ]]
    Out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    Out.dims = [6, 1]

Example 2:

Given a 1-level phi::DenseTensor input(X):
    X.lod =  [[ 0,     2,                   5      6 ]]
    X.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    X.dims = [6, 1]

input(Y) is a Tensor:
    Y.data = [[0, 2, 6]]
    Y.dims = [1, 3]

then we get a 1-level LoDTensor:
    Out.lod =  [[ 0,     2,                          6 ]]
    Out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    Out.dims = [6, 1]

Example 3:

Given a 1-level phi::DenseTensor input(X):
    X.lod =  [[ 0,      2,                   5     6 ]]
    X.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    X.dims = [6, 1]

input(Y) is a 2-level LoDTensor:
    Y.lod =  [[0, 2, 4], [0, 2, 5, 6]]
    Y.data = [[1.1], [2.1], [3.1], [4.1], [5.1], [6.1]]
    Y.dims = [6, 1]

then we get a 2-level LoDTensor:
    Out.lod =  [[0, 2, 4], [0, 2, 5, 6]]
    Out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    Out.dims = [6, 1]

)DOC");
  }
};

class LoDResetGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "LoDResetGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Output",
                   framework::GradVarName("Out"),
                   "LoDResetGrad");

    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(
                              ctx, framework::GradVarName("Out")),
                          ctx.device_context().GetPlace());
  }
};

template <typename T>
class LoDResetGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("lod_reset_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(LoDResetInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(LoDResetGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

DECLARE_NO_NEED_BUFFER_VARS_INFERER(LoDResetGradNoNeedBufferVarInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(lod_reset,
                  ops::LoDResetOp,
                  ops::LoDResetOpMaker,
                  ops::LoDResetGradMaker<paddle::framework::OpDesc>,
                  ops::LoDResetGradMaker<paddle::imperative::OpBase>,
                  ops::LoDResetOpVarTypeInference,
                  ops::LoDResetInplaceInferer);
REGISTER_OPERATOR(lod_reset_grad,
                  ops::LoDResetGradOp,
                  ops::LoDResetGradNoNeedBufferVarInferer,
                  ops::LoDResetGradInplaceInferer);

PD_REGISTER_STRUCT_KERNEL(lod_reset,
                          CPU,
                          ALL_LAYOUT,
                          ops::LoDResetKernel,
                          plat::float16,
                          float,
                          double,
                          int,
                          int64_t) {}

#ifdef PADDLE_WITH_XPU
PD_REGISTER_STRUCT_KERNEL(lod_reset,
                          XPU,
                          ALL_LAYOUT,
                          ops::LoDResetKernel,
                          plat::float16,
                          float,
                          double,
                          int,
                          int64_t) {}
#endif

PD_REGISTER_STRUCT_KERNEL(lod_reset_grad,
                          CPU,
                          ALL_LAYOUT,
                          ops::LoDResetGradKernel,
                          plat::float16,
                          float,
                          double,
                          int,
                          int64_t) {}
