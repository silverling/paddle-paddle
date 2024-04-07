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

#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/phi/core/type_defs.h"

namespace phi {

KernelSignature AdadeltaOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorInput("Grad")) {
    return KernelSignature("adadelta",
                           {"Param",
                            "Grad",
                            "AvgSquaredGrad",
                            "AvgSquaredUpdate",
                            "LearningRate",
                            "MasterParam"},
                           {"rho", "epsilon", "multi_precision"},
                           {"ParamOut",
                            "AvgSquaredGradOut",
                            "AvgSquaredUpdateOut",
                            "MasterParamOut"});
  }

  return KernelSignature("unregistered", {}, {}, {});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(adadelta, phi::AdadeltaOpArgumentMapping);
