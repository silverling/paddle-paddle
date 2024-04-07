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

#include <stdint.h>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dense_tensor.inl"

namespace paddle {
namespace framework {
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class MaxSequenceLenOp : public framework::OperatorBase {
 public:
  MaxSequenceLenOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto &rank_table =
        scope.FindVar(Input("RankTable"))->Get<framework::LoDRankTable>();
    auto *out = scope.FindVar(Output("Out"))->GetMutable<phi::DenseTensor>();
    int64_t *out_ptr = out->mutable_data<int64_t>({1}, platform::CPUPlace());
    *out_ptr = rank_table.items()[0].length;  // NOLINT
  }
};

class MaxSequenceLenOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("RankTable", "Input variable which is a LoDRankTable object");
    AddOutput("Out", "The max sequence length");
    AddComment(R"DOC(
    Given a LoDRankTable object, this layer returns the max length of
    a batch of sequences. In fact, a LoDRankTable object contains a list of
    tuples(<sequence index, sequence length>) and the list is already sorted by
    sequence length in descending order, so the operator just returns the
    sequence length of the first tuple element
)DOC");
  }
};

class MaxSequenceLenInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(
        context->HasInput("RankTable"), "Input", "RankTable", "MaxSequenceLen");
    context->SetOutputDim("Out", {1});
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    max_sequence_len,
    paddle::operators::MaxSequenceLenOp,
    paddle::operators::MaxSequenceLenOpProtoMaker,
    paddle::operators::MaxSequenceLenInferShape,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
