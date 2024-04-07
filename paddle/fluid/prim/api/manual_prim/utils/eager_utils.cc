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
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace prim {

template <>
Tensor empty<Tensor>(const paddle::experimental::IntArray& shape,
                     phi::DataType dtype,
                     const paddle::Place& place) {
  if (dtype == phi::DataType::UNDEFINED) {
    dtype = phi::DataType::FLOAT32;
  }
  return empty_ad_func(shape, dtype, place);
}

template <>
Tensor empty_like<Tensor>(const paddle::Tensor& x,
                          phi::DataType dtype,
                          const paddle::Place& place) {
  if (dtype == phi::DataType::UNDEFINED) {
    dtype = phi::DataType::FLOAT32;
  }
  return empty_like_ad_func(x, dtype, place);
}

template <>
void set_output<Tensor>(const paddle::Tensor& x_tmp, paddle::Tensor* x) {
  x->set_impl(x_tmp.impl());
  x->set_autograd_meta(x_tmp.mutable_autograd_meta());
}

template <>
void by_pass<Tensor>(const paddle::Tensor& x, Tensor* out) {
  set_output<Tensor>(x, out);
}

}  // namespace prim
}  // namespace paddle
