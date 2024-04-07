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

#include "paddle/fluid/operators/collective/c_allgather_op.h"

#include <algorithm>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/attribute_checker.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/utils/variant.h"

namespace phi {
class CPUContext;
}  // namespace phi

namespace paddle {
namespace operators {

class CAllGatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "AllGather");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Input", "Out", "AllGather");
    int nranks = ctx->Attrs().Get<int>("nranks");
    PADDLE_ENFORCE_GE(nranks,
                      2,
                      platform::errors::InvalidArgument(
                          "The value of nranks should be >=2."));
    framework::DDim dim = ctx->GetInputDim("X");
    // 0D use stack/unstack while others use concat/split
    if (dim.size() == 0) {
      dim = common::make_ddim({nranks});
    } else {
      dim[0] = dim[0] * nranks;
      if (dim[0] < 0) dim[0] = -1;
    }
    ctx->SetOutputDim("Out", dim);
  }
};

class CAllGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) tensor to be allgather");
    AddOutput("Out", "(Tensor) the allgather result");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddAttr<int>("nranks",
                 "Total trainer count of the distributed training job");
    AddComment(R"DOC(
CAllGather Operator
each rank receives the aggregation of data from all ranks in the order of the ranks

reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allgather
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_allgather,
                             ops::CAllGatherOp,
                             ops::CAllGatherOpMaker);

PD_REGISTER_STRUCT_KERNEL(c_allgather,
                          CPU,
                          ALL_LAYOUT,
                          ops::CAllGatherOpCPUKernel,
                          float,
                          double,
                          int,
                          int8_t,
                          int64_t,
                          uint8_t,
                          bool,
                          plat::float16) {}
