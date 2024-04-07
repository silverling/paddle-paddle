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

#include <stddef.h>

#include "paddle/fluid/pir/dialect/kernel/ir/type_storage.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/common/ddim.h"
#include "paddle/common/layout.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/storage_manager_support.h"
#include "paddle/pir/include/core/type_base.h"
#include "paddle/pir/include/core/type_id.h"

namespace pir {
class IrContext;
class WrapTypeInterface;
}  // namespace pir

namespace paddle {
namespace dialect {
struct AllocatedDenseTensorArrayTypeStorage;
struct AllocatedDenseTensorTypeStorage;
struct AllocatedSelectedRowsTypeStorage;

class AllocatedDenseTensorType
    : public pir::Type::TypeBase<AllocatedDenseTensorType,
                                 pir::Type,
                                 AllocatedDenseTensorTypeStorage,
                                 pir::WrapTypeInterface> {
 public:
  using Base::Base;

  static AllocatedDenseTensorType get(pir::IrContext *ctx,
                                      const phi::Place &place,
                                      dialect::DenseTensorType type) {
    return pir::TypeManager::template get<AllocatedDenseTensorType>(
        ctx, place, type);
  }

  static AllocatedDenseTensorType get(pir::IrContext *ctx,
                                      const phi::Place &place,
                                      const pir::Type &dtype,
                                      const phi::DDim &dims,
                                      const phi::DataLayout &layout,
                                      const phi::LoD &lod,
                                      size_t offset) {
    dialect::DenseTensorType dense_tensor_type =
        dialect::DenseTensorType::get(ctx, dtype, dims, layout, lod, offset);

    return pir::TypeManager::template get<AllocatedDenseTensorType>(
        ctx, place, dense_tensor_type);
  }

  pir::Type prim_type();

  const phi::Place &place() const;

  pir::Type dtype() const;

  const phi::DDim &dims() const;

  phi::DataLayout data_layout() const;

  const phi::LoD &lod() const;

  size_t offset() const;
};

class AllocatedSelectedRowsType
    : public pir::Type::TypeBase<AllocatedSelectedRowsType,
                                 pir::Type,
                                 AllocatedSelectedRowsTypeStorage,
                                 pir::WrapTypeInterface> {
 public:
  using Base::Base;

  static AllocatedSelectedRowsType get(pir::IrContext *ctx,
                                       const phi::Place &place,
                                       dialect::SelectedRowsType type) {
    return pir::TypeManager::template get<AllocatedSelectedRowsType>(
        ctx, place, type);
  }

  static AllocatedSelectedRowsType get(pir::IrContext *ctx,
                                       const phi::Place &place,
                                       const pir::Type &dtype,
                                       const phi::DDim &dims,
                                       const phi::DataLayout &layout,
                                       const phi::LoD &lod,
                                       size_t offset) {
    dialect::SelectedRowsType type =
        dialect::SelectedRowsType::get(ctx, dtype, dims, layout, lod, offset);

    return pir::TypeManager::template get<AllocatedSelectedRowsType>(
        ctx, place, type);
  }

  pir::Type prim_type();

  const phi::Place &place() const;

  pir::Type dtype() const;

  const phi::DDim &dims() const;

  phi::DataLayout data_layout() const;

  const phi::LoD &lod() const;

  size_t offset() const;
};

class AllocatedDenseTensorArrayType
    : public pir::Type::TypeBase<AllocatedDenseTensorArrayType,
                                 pir::Type,
                                 AllocatedDenseTensorArrayTypeStorage,
                                 pir::WrapTypeInterface> {
 public:
  using Base::Base;

  static AllocatedDenseTensorArrayType get(pir::IrContext *ctx,
                                           const phi::Place &place,
                                           dialect::DenseTensorArrayType type) {
    return pir::TypeManager::template get<AllocatedDenseTensorArrayType>(
        ctx, place, type);
  }

  static AllocatedDenseTensorArrayType get(pir::IrContext *ctx,
                                           const phi::Place &place,
                                           const pir::Type &dtype,
                                           const phi::DDim &dims,
                                           const phi::DataLayout &layout) {
    dialect::DenseTensorArrayType type =
        dialect::DenseTensorArrayType::get(ctx, dtype, dims, layout);

    return pir::TypeManager::template get<AllocatedDenseTensorArrayType>(
        ctx, place, type);
  }

  pir::Type prim_type();

  const phi::Place &place() const;

  const pir::Type &dtype() const;

  const pir::DDim &dims() const;

  const phi::DataLayout &data_layout() const;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedDenseTensorType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedSelectedRowsType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedDenseTensorArrayType)
