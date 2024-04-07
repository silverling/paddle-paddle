// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/phi/core/type_defs.h"
#include "paddle/utils/any.h"

namespace phi {

KernelSignature EmbeddingGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorInput("W")) {
    if ((paddle::any_cast<bool>(ctx.Attr("is_sparse"))) == true) {
      return KernelSignature("embedding_sparse_grad",
                             {"Ids", "W", "Out@GRAD"},
                             {"padding_idx"},
                             {"W@GRAD"});
    } else {
      return KernelSignature("embedding_grad",
                             {"Ids", "W", "Out@GRAD"},
                             {"padding_idx"},
                             {"W@GRAD"});
    }
  } else {
    if ((paddle::any_cast<bool>(ctx.Attr("is_sparse"))) == true) {
      return KernelSignature("sparse_weight_embedding_sparse_grad",
                             {"Ids", "W", "Out@GRAD"},
                             {"padding_idx"},
                             {"W@GRAD"});
    } else {
      return KernelSignature("sparse_weight_embedding_grad",
                             {"Ids", "W", "Out@GRAD"},
                             {"padding_idx"},
                             {"W@GRAD"});
    }
  }
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(lookup_table_v2_grad, embedding_grad);
PD_REGISTER_BASE_KERNEL_NAME(lookup_table_v2_grad, embedding_sparse_grad);
PD_REGISTER_BASE_KERNEL_NAME(lookup_table_v2_grad,
                             sparse_weight_embedding_grad);
PD_REGISTER_BASE_KERNEL_NAME(lookup_table_v2_grad,
                             sparse_weight_embedding_sparse_grad);

PD_REGISTER_ARG_MAPPING_FN(lookup_table_v2_grad,
                           phi::EmbeddingGradOpArgumentMapping);
