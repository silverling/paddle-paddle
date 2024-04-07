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

#pragma once

#include <memory>
#include <vector>

#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/global_and_sub_mesh_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/nd_mesh_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/p_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/r_to_x_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_p_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/s_to_s_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/same_status_reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/x_to_r_reshard_function.h"

namespace phi {
namespace distributed {
class DistTensor;
class TensorDistAttr;

std::vector<std::unique_ptr<ReshardFunction>>& GetReshardFunctionList();

#define REGISTER_RESHARD_FUNC(func_type)                                    \
  class __RegisterReshard_##func_type {                                     \
   public:                                                                  \
    __RegisterReshard_##func_type() {                                       \
      GetReshardFunctionList().emplace_back(std::make_unique<func_type>()); \
    }                                                                       \
  };                                                                        \
  static __RegisterReshard_##func_type local_reshard_func_##func_type

ReshardFunction* ChooseProperReshardFunction(
    const DistTensor& in, const TensorDistAttr& out_dist_attr);

}  // namespace distributed
}  // namespace phi
