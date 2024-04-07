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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"
#include "paddle/pir/include/dialect/shape/utils/shape_or_data_expr.h"

namespace pir {
class ShapeConstraintIRAnalysis;
class Value;
}  // namespace pir

inline bool GetBoolAttr(const pir::Operation *op, const std::string &str) {
  const auto &attr_map = op->attributes();
  PADDLE_ENFORCE(
      attr_map.count(str),
      phi::errors::PreconditionNotMet(
          "attr [%s] MUST in attribute map for [%s] op", str, op->name()));
  return attr_map.at(str).dyn_cast<pir::BoolAttribute>().data();
}

// To make codes shorter
using ExprVec = std::vector<symbol::DimExpr>;
using ShapeOrData = symbol::ShapeOrDataDimExprs;
using TensorExprs = symbol::TensorShapeOrDataDimExprs;
using TensorListExprs = symbol::TensorListShapeOrDataDimExprs;

namespace paddle::dialect::details {
template <typename T>
struct AttributeTrait;

template <>
struct AttributeTrait<std::int64_t> {
  using value_type = ::pir::Int64Attribute;
};

template <>
struct AttributeTrait<int> {
  using value_type = ::pir::Int32Attribute;
};

template <typename T = int64_t>
std::vector<T> GetVectorAttr(const ::pir::Operation *op,
                             const std::string &name) {
  using value_type = typename AttributeTrait<T>::value_type;

  const auto &attr_map = op->attributes();
  PADDLE_ENFORCE(
      attr_map.count(name),
      phi::errors::PreconditionNotMet(
          "attr [%s] MUST in attribute map for [%s] op", name, op->name()));
  const auto &val = attr_map.at(name);

  PADDLE_ENFORCE(val.isa<::pir::ArrayAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "axis Type MUST ArrayAttribute for [%s] op", op->name()));
  auto array_list = val.dyn_cast<::pir::ArrayAttribute>().AsVector();
  std::vector<T> vec_res;
  if (array_list.size() > 0) {
    PADDLE_ENFORCE_EQ(array_list[0].isa<value_type>(),
                      true,
                      phi::errors::Unimplemented(
                          "the 0th elementwise MUST be ir::Int64Attribute"));
    for (size_t i = 0; i < array_list.size(); ++i) {
      vec_res.push_back(array_list[i].dyn_cast<value_type>().data());
    }
  }
  return vec_res;
}

inline ExprVec GetExprVecFromData(const ShapeOrData &shapeordata) {
  if (shapeordata.isa<TensorListExprs>()) {
    ExprVec result;
    TensorListExprs list =
        shapeordata.dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
    for (size_t i = 0; i < list.size(); i++) {
      for (auto expr : list[i].data().value()) {
        result.emplace_back(expr);
      }
    }
    return result;
  } else {
    return shapeordata.data().value();
  }
}

inline ExprVec GetExprVecFromShape(const ShapeOrData &shapeordata) {
  const auto GetShapeExprsFromList = [&]() {
    ExprVec result;
    TensorListExprs list =
        shapeordata.dyn_cast<symbol::TensorListShapeOrDataDimExprs>();
    for (size_t i = 0; i < list.size(); i++) {
      for (auto expr : list[i].data().value()) {
        result.emplace_back(expr);
      }
    }
    return result;
  };
  if (shapeordata.isa<TensorListExprs>()) {
    return GetShapeExprsFromList();
  } else {
    return shapeordata.shape();
  }
}

std::optional<std::vector<int64_t>> VecExpr2Int64(const ExprVec &expr_vec);

ExprVec VecInt642Expr(const std::vector<int64_t> &int_vec);

bool ReduceInferDim(pir::Operation *op,
                    pir::ShapeConstraintIRAnalysis *shape_analysis,
                    const std::vector<int64_t> &axis,
                    bool keep_dim,
                    bool reduce_all);

void BuildCstrEqForTensorListAlongAxis(
    pir::ShapeConstraintIRAnalysis *shape_analysis,
    const symbol::TensorListShapeOrDataDimExprs &shape_data_list,
    int axis);

void BuildCstrEqForTensorListAlongAxis(
    pir::ShapeConstraintIRAnalysis *shape_analysis,
    const std::vector<pir::Value> &values,
    int axis);

}  // namespace paddle::dialect::details
