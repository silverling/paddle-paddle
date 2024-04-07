/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/strings/unary.h"

#include "paddle/common/layout.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {
namespace strings {

void UnchangedInferMeta(const StringTensorMeta& x_meta, MetaTensor* out) {
  out->set_dims(x_meta.dims);
  out->set_dtype(DataType::PSTRING);
  out->set_layout(DataLayout::PSTRING_UNION);
}

void CreateLikeInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

}  // namespace strings
}  // namespace phi
