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

#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/core/compat/get_kerneltype_forvar_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/attribute.h"

namespace phi {
class DenseTensor;
class KernelKey;
}  // namespace phi

namespace paddle {
namespace framework {

class OpKernelType;
class Variable;

void TransformData(const phi::KernelKey &expected_kernel_type,
                   const phi::KernelKey &kernel_type_for_var,
                   const phi::DenseTensor &input_tensor,
                   phi::DenseTensor *out,
                   const phi::Place &place);

/**
 * Set OutVar from InVar, except the tensor is shared with `tensor`
 */
void SetTensorToVariable(const Variable &in_var,
                         const phi::DenseTensor &tensor,
                         Variable *out_var);

phi::GetKernelTypeForVarContext BuildGetKernelTypeForVarContext(
    const phi::KernelKey &kernel_key,
    const AttributeMap &fluid_attrs,
    phi::AttributeMap *phi_attrs,
    bool has_infer_varkernel_fn);

}  // namespace framework
}  // namespace paddle
