/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/dequantize_abs_max_op.h"

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_registry.h"

namespace paddle {
namespace framework {
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

template <typename T>
struct DequantizeFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& dev_ctx,
                  const phi::DenseTensor* in,
                  const phi::DenseTensor* scale,
                  float max_range,
                  phi::DenseTensor* out) {
    const float* scale_factor = scale->data<float>();
    const T* input_data = in->data<T>();
    float* output_data = out->mutable_data<float>(dev_ctx.GetPlace());
    int ind = static_cast<int>(in->numel());
    for (size_t i = 0; i < (unsigned)ind; i++) {
      output_data[i] = scale_factor[0] * input_data[i] / max_range;
    }
  }
};

template struct DequantizeFunctor<phi::CPUContext, int8_t>;
template struct DequantizeFunctor<phi::CPUContext, int16_t>;

class DequantizeMaxAbsOp : public framework::OperatorWithKernel {
 public:
  DequantizeMaxAbsOp(const std::string& type,
                     const framework::VariableNameMap& inputs,
                     const framework::VariableNameMap& outputs,
                     const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "DequantizeMaxAbs");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "DequantizeMaxAbs");

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return phi::KernelKey(data_type, ctx.device_context().GetPlace());
  }
};

class DequantizeMaxAbsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Int Tensor) The input with int8/16 type is the "
             "low precision tensor.");
    AddInput("Scale", "(float) The scale in quantization stage.");
    AddOutput("Out",
              "(float32 Tensor) The output is the dequantized high "
              "precision tensor.");
    AddAttr<float>("max_range", "(float) The max range in quantization stage.");
    AddComment(R"DOC(
DequantizeMaxAbsOp operator.

This calculation is an opposite operation of QuantizeMaxAbsOp:

$$Out = \frac{scale*X}{ max\_range }$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    dequantize_abs_max,
    ops::DequantizeMaxAbsOp,
    ops::DequantizeMaxAbsOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

PD_REGISTER_STRUCT_KERNEL(dequantize_abs_max,
                          CPU,
                          ALL_LAYOUT,
                          ops::DequantizeMaxAbsKernel,
                          int8_t,
                          int16_t) {}
