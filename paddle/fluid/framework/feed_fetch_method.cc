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

#include "paddle/fluid/framework/feed_fetch_method.h"

#include <string>
#include <ostream>
#include <typeinfo>

#include "glog/logging.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/phi_tensor_base_vector.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/string_array.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dense_tensor.inl"
#include "paddle/phi/core/enforce.h"
#include "paddle/utils/variant.h"

COMMON_DECLARE_bool(enable_pir_in_executor);
COMMON_DECLARE_bool(enable_pir_api);

namespace paddle {
namespace framework {

void SetVariable(Scope* scope,
                 const phi::DenseTensor& input,
                 const std::string& var_name) {
  Variable* target_var = scope->FindVar(var_name);
  if (target_var && !target_var->IsType<phi::DenseTensor>()) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The variable you want to set is not a phi::DenseTensor, but here "
        "you tried to convert its type to phi::DenseTensor."));
  }
  target_var = scope->Var(var_name);
  auto tensor = target_var->GetMutable<phi::DenseTensor>();
  tensor->ShareDataWith(input);
  tensor->set_lod(input.lod());
}

void SetFeedVariable(Scope* scope,
                     const phi::DenseTensor& input,
                     const std::string& var_name,
                     size_t index) {
  // If var_name Variable is not found in GlobalScope, a new variable will
  // be created.
  VLOG(3) << "SetFeedVariable name=" << var_name << " index=" << index;
  if (FLAGS_enable_pir_in_executor) {
    // shared data with input tensor
    auto feed_ele = scope->Var(var_name);
    if (!feed_ele->IsType<phi::DenseTensor>()) {
      VLOG(3) << "Reset " << var_name << " to phi::DenseTensor";
      feed_ele->Clear();
    }
    auto val = feed_ele->GetMutable<phi::DenseTensor>();
    val->ShareDataWith(input);
    // set lod
    val->set_lod(input.lod());
  } else {
    Variable* g_feed_value = scope->Var(var_name);
    auto& feed_inputs = *(g_feed_value->GetMutable<FeedList>());
    if (index >= feed_inputs.size()) {
      feed_inputs.resize(index + 1);
    }
    // shared data with input tensor
    auto& val = PADDLE_GET(phi::DenseTensor, feed_inputs[index]);
    val.ShareDataWith(input);
    // set lod
    val.set_lod(input.lod());
  }
}

void SetFeedVariable(Scope* scope,
                     const std::vector<std::string>& input,
                     const std::string& var_name,
                     size_t index) {
  // If var_name Variable is not found in GlobalScope, a new variable will
  // be created.
  VLOG(3) << "SetFeedStringVariable name=" << var_name << " index=" << index;
  Variable* g_feed_value = scope->Var(var_name);
  auto& feed_inputs = *(g_feed_value->GetMutable<FeedList>());
  if (index >= feed_inputs.size()) {
    feed_inputs.resize(index + 1);
  }
  // shared data with input tensor
  feed_inputs[index] = Strings(input);
}

FetchType& GetFetchVariable(const Scope& scope,
                            const std::string& var_name,
                            size_t index) {
  // Since we want to fetch FetchType from a variable, the variable must
  // be created already.
  Variable* g_fetch_value = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(g_fetch_value,
                          platform::errors::NotFound(
                              "Variable %s is not found in scope.", var_name));
  PADDLE_ENFORCE_EQ(g_fetch_value->IsType<FetchList>(),
                    true,
                    platform::errors::InvalidArgument(
                        "Only %s can be invoked by GetFetchVariable",
                        typeid(FetchList).name()));
  auto& fetch_outputs = *g_fetch_value->GetMutable<FetchList>();
  auto& tensor = fetch_outputs[index];
  VLOG(3) << "Fetch " << var_name << " with index " << index;
  PADDLE_ENFORCE_LT(index,
                    fetch_outputs.size(),
                    platform::errors::InvalidArgument(
                        "index must less than fetch_outputs size."));
  return tensor;
}

phi::DenseTensor& GetVariableTensor(const Scope& scope,
                                    const std::string& var_name) {
  Variable* var = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(var,
                          platform::errors::NotFound(
                              "Variable %s is not found in scope.", var_name));
  PADDLE_ENFORCE_EQ(var->IsType<phi::DenseTensor>(),
                    true,
                    platform::errors::InvalidArgument(
                        "Only support DenseTensor in GetVariableTensor now."));
  return *var->GetMutable<phi::DenseTensor>();
}

}  // namespace framework
}  // namespace paddle
