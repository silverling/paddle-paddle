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
#include <list>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "gtest/gtest-message.h"
#include "gtest/gtest-test-part.h"
#include "gtest/gtest_pred_impl.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dense_tensor.inl"

#define Conn(x, y) x##y

namespace f = paddle::framework;
namespace p = paddle::platform;

template <typename T>
void Compare1(f::Scope* scope,
              const p::DeviceContext& ctx,
              std::string op_type) {
  // init
  auto var_x = scope->Var("tmp");
  auto x = var_x->GetMutable<phi::DenseTensor>();
  std::vector<T> main_x = {1.0};
  paddle::framework::TensorFromVector(main_x, ctx, x);

  auto var_id = scope->Var("Id");
  auto id = var_id->GetMutable<phi::DenseTensor>();
  std::vector<int64_t> main_id = {1};
  paddle::framework::TensorFromVector(main_id, ctx, id);
  for (int i = 0; i < 3; i++) {
    auto& child_scope = scope->NewScope();
    auto child_var = child_scope.Var("tmp");
    auto tensor_x = child_var->GetMutable<phi::DenseTensor>();
    std::vector<T> init_x = {static_cast<T>(i)};
    paddle::framework::TensorFromVector(init_x, ctx, tensor_x);
  }

  ctx.Wait();

  // run
  f::AttributeMap attrs = {{"to_main_scope", false}, {"num_micro_batches", 3}};
  f::VariableNameMap output;
  auto op = f::OpRegistry::CreateOp(
      op_type, {{"X", {"tmp"}}, {"Id", {"Id"}}}, output, attrs);

  auto place = ctx.GetPlace();
  op->Run(*scope, place);
  ctx.Wait();

  std::list<f::Scope*>::const_iterator iter = scope->kids().begin();
  iter++;
  iter++;

  auto* kid_scope = *iter;
  auto* dst_var = kid_scope->FindVar("tmp");
  auto* tensor_out = dst_var->GetMutable<phi::DenseTensor>();

  std::vector<T> out_vec;
  paddle::framework::TensorToVector(*tensor_out, ctx, &out_vec);

  int expected = 1;
  EXPECT_EQ(static_cast<int>(out_vec[0]), expected);
}

template <typename T>
void Compare2(f::Scope* scope,
              const p::DeviceContext& ctx,
              std::string op_type) {
  // init
  auto var_x = scope->Var("tmp");
  auto x = var_x->GetMutable<phi::DenseTensor>();
  std::vector<T> main_x = {1.0};
  paddle::framework::TensorFromVector(main_x, ctx, x);

  auto var_id = scope->Var("Id");
  auto id = var_id->GetMutable<phi::DenseTensor>();
  std::vector<int64_t> main_id = {0};
  paddle::framework::TensorFromVector(main_id, ctx, id);
  for (int i = 0; i < 3; i++) {
    auto& child_scope = scope->NewScope();
    auto child_var = child_scope.Var("tmp");
    auto tensor_x = child_var->GetMutable<phi::DenseTensor>();
    std::vector<T> init_x = {static_cast<T>(i)};
    paddle::framework::TensorFromVector(init_x, ctx, tensor_x);
  }

  ctx.Wait();

  // run
  f::AttributeMap attrs = {{"to_main_scope", true}, {"num_micro_batches", 3}};
  f::VariableNameMap output;
  auto op = f::OpRegistry::CreateOp(
      op_type, {{"X", {"tmp"}}, {"Id", {"Id"}}}, output, attrs);

  auto place = ctx.GetPlace();
  op->Run(*scope, place);
  ctx.Wait();

  auto* dst_var = scope->FindVar("tmp");
  auto* tensor_out = dst_var->GetMutable<phi::DenseTensor>();

  std::vector<T> out_vec;
  paddle::framework::TensorToVector(*tensor_out, ctx, &out_vec);

  int expected = 0;
  EXPECT_EQ(static_cast<int>(out_vec[0]), expected);
}

#ifdef PADDLE_WITH_CUDA
TEST(copy_cross_scope, CUDA_fp32) {
  f::Scope scope;
  phi::GPUContext ctx(p::CUDAPlace(0));
  ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                       .GetAllocator(p::CUDAPlace(0), ctx.stream())
                       .get());
  ctx.PartialInitWithAllocator();
  Compare1<float>(&scope, ctx, "copy_cross_scope");
}

TEST(copy_cross_scope_to_main_scope, CUDA_fp32) {
  f::Scope scope;
  phi::GPUContext ctx(p::CUDAPlace(0));
  ctx.SetAllocator(paddle::memory::allocation::AllocatorFacade::Instance()
                       .GetAllocator(p::CUDAPlace(0), ctx.stream())
                       .get());
  ctx.PartialInitWithAllocator();
  Compare2<float>(&scope, ctx, "copy_cross_scope");
}
#endif
