//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/attribute_checker.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/utils/variant.h"

namespace paddle {
namespace framework {
class OpDesc;
template <typename T> class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative

namespace operators {
class ElementwiseOp;
class ElementwiseOpInferVarType;

class FusedElementwiseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInput("X", "The first input tensor of elementwise op.");
    AddInput("Y", "The second input tensor of elementwise op.");
    AddOutput("Out", "A location into which the result is stored.");
    AddAttr<int>(
        "axis",
        "If X.dimension != Y.dimension, Y.dimension must be a "
        "subsequence of X.dimension. And axis is the start dimension index "
        "for broadcasting Y onto X.")
        .SetDefault(-1);
    AddAttr<std::string>(
        "fuse_activation",
        "Activation type from elementwise_act_onednn_fuse_pass")
        .SetDefault("");
    AddAttr<float>("fuse_alpha",
                   "Activation alpha from elementwise_act_onednn_fuse_pass")
        .SetDefault(0.0f);
    AddAttr<float>("fuse_beta",
                   "Activation beta from elementwise_act_onednn_fuse_pass")
        .SetDefault(0.0f);
    AddAttr<float>("fused_output_scale",
                   "Obtained from operator_scale_onednn_fuse_pass")
        .SetDefault(1.0f);
    AddAttr<std::vector<int>>(
        "fused_unsqueeze2_axes",
        "Obtained from operator_unsqueeze2_onednn_fuse_pass")
        .SetDefault({});
    AddAttr<float>("scale_x", "Elementwise X input quantization scale")
        .SetDefault(1.0f);
    AddAttr<float>("scale_y", "Elementwise Y input quantization scale")
        .SetDefault(1.0f);
    AddAttr<float>("scale_out", "Elementwise Out output quantization scale")
        .SetDefault(1.0f);
    AddComment(
        R"DOC(Elementwise operator extended with oneDNN-specific fusion logic.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_elementwise_add,
    ops::ElementwiseOp,
    ops::FusedElementwiseOpMaker,
    ops::ElementwiseOpInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    fused_elementwise_sub,
    ops::ElementwiseOp,
    ops::FusedElementwiseOpMaker,
    ops::ElementwiseOpInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    fused_elementwise_mul,
    ops::ElementwiseOp,
    ops::FusedElementwiseOpMaker,
    ops::ElementwiseOpInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    fused_elementwise_div,
    ops::ElementwiseOp,
    ops::FusedElementwiseOpMaker,
    ops::ElementwiseOpInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
