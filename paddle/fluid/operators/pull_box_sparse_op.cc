//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/pull_box_sparse_op.h"

#include <ext/alloc_traits.h>
#include <stdint.h>
#include <algorithm>
#include <string>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/attribute_checker.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/grad_op_desc_maker.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/utils/variant.h"

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

class PullBoxSparseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GE(
        ctx->Inputs("Ids").size(),
        1UL,
        platform::errors::InvalidArgument(
            "Inputs(Ids) of PullBoxSparseOp should not be empty."));
    PADDLE_ENFORCE_GE(
        ctx->Outputs("Out").size(),
        1UL,
        platform::errors::InvalidArgument(
            "Outputs(Out) of PullBoxSparseOp should not be empty."));
    auto hidden_size = static_cast<int64_t>(ctx->Attrs().Get<int>("size"));
    auto all_ids_dim = ctx->GetInputsDim("Ids");
    const size_t n_ids = all_ids_dim.size();
    std::vector<framework::DDim> outs_dims;
    outs_dims.resize(n_ids);
    for (size_t i = 0; i < n_ids; ++i) {
      const auto ids_dims = all_ids_dim[i];
      int ids_rank = ids_dims.size();
      PADDLE_ENFORCE_EQ(ids_dims[ids_rank - 1],
                        1,
                        platform::errors::InvalidArgument(
                            "Shape error in %lu id, the last dimension of the "
                            "'Ids' tensor must be 1.",
                            i));
      auto out_dim =
          common::vectorize(common::slice_ddim(ids_dims, 0, ids_rank - 1));
      out_dim.push_back(hidden_size);
      outs_dims[i] = common::make_ddim(out_dim);
    }
    ctx->SetOutputsDim("Out", outs_dims);
    for (size_t i = 0; i < n_ids; ++i) {
      ctx->ShareLoD("Ids", "Out", i, i);
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
  }
};

class PullBoxSparseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.")
        .AsDispensable();
    AddInput("Ids",
             "Input tensors with type int32 or int64 "
             "contains the ids to be looked up in BoxPS. "
             "The last dimension size must be 1.")
        .AsDuplicable();
    AddOutput("Out", "The lookup results tensors.").AsDuplicable();
    AddAttr<bool>("is_sparse",
                  "(boolean, default false) "
                  "Sparse update.")
        .SetDefault(false);
    AddAttr<bool>("is_distributed",
                  "(boolean, default false) distributed lookup table.")
        .SetDefault(false);
    AddAttr<int>("size", "(int, the embedding hidden size").SetDefault(1);
    AddComment(R"DOC(
Pull Box Sparse Operator.

This operator is used to perform lookups on the BoxPS,
then concatenated into a dense tensor.

The input Ids can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.

)DOC");
  }
};

template <typename T>
class PushBoxSparseOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("push_box_sparse");
    op->SetInput("Ids", this->Input("Ids"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
  }
};

class PushBoxSparseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(
                              ctx, framework::GradVarName("Out")),
                          ctx.GetPlace());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(pull_box_sparse,
                  ops::PullBoxSparseOp,
                  ops::PullBoxSparseOpMaker,
                  ops::PushBoxSparseOpMaker<paddle::framework::OpDesc>,
                  ops::PushBoxSparseOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(push_box_sparse, ops::PushBoxSparseOp);

PD_REGISTER_STRUCT_KERNEL(
    pull_box_sparse, CPU, ALL_LAYOUT, ops::PullBoxSparseKernel, float) {}
PD_REGISTER_STRUCT_KERNEL(
    push_box_sparse, CPU, ALL_LAYOUT, ops::PushBoxSparseKernel, float) {}
