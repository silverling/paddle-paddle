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

#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute_storage.h"
#include "paddle/pir/include/core/attribute_base.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/type_id.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace pir {
class IrContext;
class Operation;
}  // namespace pir

namespace pir::shape {

class IR_API SymbolAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(SymbolAttribute, SymbolAttributeStorage);

  symbol::ShapeOrDataDimExprs data() const;

  static SymbolAttribute get(IrContext* ctx,
                             const symbol::ShapeOrDataDimExprs& value);

  static const char attr_name[];
};

void SetShapeAttrForOp(pir::Operation* op,
                       const symbol::ShapeOrDataDimExprs& shape_data);

}  // namespace pir::shape

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::shape::SymbolAttribute)
