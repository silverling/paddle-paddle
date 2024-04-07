// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/distributed/ir/dist_op.h"

#include <ostream>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "glog/logging.h"
#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/type.h"

namespace pir {
class Builder;
}  // namespace pir

namespace paddle {
namespace dialect {

const char* ShardTensorOp::attributes_name[1] = {"op_dist_attr"};
const char* ReShardOp::attributes_name[1] = {"op_dist_attr"};

void ShardTensorOp::VerifySig() {
  VLOG(4)
      << "Start Verifying inputs, outputs and attributes for: ShardTensorOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        1u,
        common::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 1.", input_size));
    PADDLE_ENFORCE_EQ((*this)
                          ->operand_source(0)
                          .type()
                          .isa<paddle::dialect::DenseTensorType>(),
                      true,
                      common::errors::PreconditionNotMet(
                          "Type validation failed for the 0th input."));
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto& attributes = this->attributes();
    PADDLE_ENFORCE_EQ((attributes.count("op_dist_attr") > 0 &&
                       attributes.at("op_dist_attr")
                           .isa<paddle::dialect::OperationDistAttribute>()),
                      true,
                      common::errors::PreconditionNotMet(
                          "Type of attribute: op_dist_attr is not right."));
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        common::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE_EQ(
        (*this)->result(0).type().isa<paddle::dialect::DistDenseTensorType>(),
        true,
        common::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }

  VLOG(4) << "Verifying op dist attrs:";
  {
    auto op_dist_attr =
        this->attribute<paddle::dialect::OperationDistAttribute>(
            "op_dist_attr");
    PADDLE_ENFORCE_EQ(op_dist_attr.num_operand_dist_attrs(),
                      0u,
                      common::errors::PreconditionNotMet(
                          "The op_dist_attr input size %d must be equal to 0.",
                          op_dist_attr.num_operand_dist_attrs()));

    PADDLE_ENFORCE_EQ(op_dist_attr.num_result_dist_attrs(),
                      num_results(),
                      common::errors::PreconditionNotMet(
                          "The op_dist_attr output size %d must "
                          "be equal to op output size %d.",
                          op_dist_attr.num_result_dist_attrs(),
                          num_results()));
  }
  VLOG(4) << "End Verifying for: ShardTensorOp.";
}

void ShardTensorOp::Build(pir::Builder& builder,
                          pir::OperationArgument& argument,
                          pir::Value input,
                          pir::AttributeMap attributes) {
  VLOG(4) << "Start build ShardOp";

  // Temporary restriction, will support input use_empty false in the future
  PADDLE_ENFORCE_EQ(
      input.use_empty(),
      true,
      common::errors::PreconditionNotMet("'input' use_empty is not true"));

  paddle::dialect::DenseTensorType input_tensor_type;
  if (input.type().isa<paddle::dialect::DenseTensorType>()) {
    input_tensor_type =
        input.type().dyn_cast<paddle::dialect::DenseTensorType>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support paddle::dialect::DenseTensorType"));
  }

  PADDLE_ENFORCE_NE(
      attributes.find("tensor_dist_attr"),
      attributes.end(),
      common::errors::NotFound(
          "'tensor_dist_attr' Attribute is expected for ShardOp"));
  paddle::dialect::TensorDistAttribute tensor_dist_attr =
      attributes.at("tensor_dist_attr")
          .dyn_cast<paddle::dialect::TensorDistAttribute>();

  VLOG(4) << "Builder construction inputs";
  argument.AddInput(input);

  VLOG(4) << "Builder construction attributes";
  auto process_mesh_attr = tensor_dist_attr.process_mesh_attr();
  auto dims_mapping = tensor_dist_attr.dims_mapping();

  pir::Attribute op_dist_attr = OperationDistAttribute::get(
      pir::IrContext::Instance(),
      process_mesh_attr,
      std::vector<TensorDistAttribute>(),
      std::vector<TensorDistAttribute>{tensor_dist_attr});
  argument.AddAttribute("op_dist_attr", op_dist_attr);

  VLOG(4) << "Builder construction outputs";
  auto global_dims = input_tensor_type.dims();
  auto process_mesh_shape = process_mesh_attr.shape();
  PADDLE_ENFORCE_EQ(static_cast<int>(dims_mapping.size()),
                    global_dims.size(),
                    common::errors::PreconditionNotMet(
                        "dims_mapping size %d does not match input size %d",
                        dims_mapping.size(),
                        global_dims.size()));
  auto local_shape = InferLocalDDim(global_dims, tensor_dist_attr);
  pir::Type out_dist_tensor_type =
      paddle::dialect::DistDenseTensorType::get(pir::IrContext::Instance(),
                                                input_tensor_type,
                                                tensor_dist_attr,
                                                local_shape);
  argument.AddOutput(out_dist_tensor_type);
  ::pir::PassStopGradientsDefaultly(argument);
}

void ReShardOp::VerifySig() {
  VLOG(4) << "Start Verifying inputs, outputs and attributes for: ReShardOp.";
  VLOG(4) << "Verifying inputs:";
  {
    auto input_size = num_operands();
    PADDLE_ENFORCE_EQ(
        input_size,
        1u,
        common::errors::PreconditionNotMet(
            "The size %d of inputs must be equal to 1.", input_size));
    PADDLE_ENFORCE_EQ((*this)
                          ->operand_source(0)
                          .type()
                          .isa<paddle::dialect::DistDenseTensorType>(),
                      true,
                      common::errors::PreconditionNotMet(
                          "Type validation failed for the 0th input."));
  }
  VLOG(4) << "Verifying attributes:";
  {
    auto& attributes = this->attributes();
    PADDLE_ENFORCE_EQ((attributes.count("op_dist_attr") > 0 &&
                       attributes.at("op_dist_attr")
                           .isa<paddle::dialect::OperationDistAttribute>()),
                      true,
                      common::errors::PreconditionNotMet(
                          "Type of attribute: op_dist_attr is not right."));
  }
  VLOG(4) << "Verifying outputs:";
  {
    auto output_size = num_results();
    PADDLE_ENFORCE_EQ(
        output_size,
        1u,
        common::errors::PreconditionNotMet(
            "The size %d of outputs must be equal to 1.", output_size));
    PADDLE_ENFORCE_EQ(
        (*this)->result(0).type().isa<paddle::dialect::DistDenseTensorType>(),
        true,
        common::errors::PreconditionNotMet(
            "Type validation failed for the 0th output."));
  }

  VLOG(4) << "Verifying op dist attrs:";
  {
    auto op_dist_attr =
        this->attribute<paddle::dialect::OperationDistAttribute>(
            "op_dist_attr");
    PADDLE_ENFORCE_EQ(op_dist_attr.num_operand_dist_attrs(),
                      1u,
                      common::errors::PreconditionNotMet(
                          "The op_dist_attr input size %d must be equal to 1.",
                          op_dist_attr.num_operand_dist_attrs()));

    PADDLE_ENFORCE_EQ(op_dist_attr.num_result_dist_attrs(),
                      num_results(),
                      common::errors::PreconditionNotMet(
                          "The op_dist_attr output size %d must "
                          "be equal to op output size %d.",
                          op_dist_attr.num_result_dist_attrs(),
                          num_results()));
  }
  VLOG(4) << "End Verifying for: ShardTensorOp.";
}

void ReShardOp::Build(pir::Builder& builder,
                      pir::OperationArgument& argument,
                      pir::Value input,
                      TensorDistAttribute tensor_dist_attr) {
  VLOG(4) << "Start build ReShardOp";

  paddle::dialect::DistDenseTensorType input_tensor_type;
  if (input.type().isa<paddle::dialect::DistDenseTensorType>()) {
    input_tensor_type =
        input.type().dyn_cast<paddle::dialect::DistDenseTensorType>();
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support paddle::dialect::DistDenseTensorType"));
  }

  VLOG(4) << "Builder construction inputs";
  argument.AddInput(input);

  VLOG(4) << "Builder construction attributes";
  pir::Attribute op_dist_attr = OperationDistAttribute::get(
      pir::IrContext::Instance(),
      input_tensor_type.tensor_dist_attr().process_mesh_attr(),
      std::vector<TensorDistAttribute>{input_tensor_type.tensor_dist_attr()},
      std::vector<TensorDistAttribute>{tensor_dist_attr});
  argument.AddAttribute("op_dist_attr", op_dist_attr);

  VLOG(4) << "Builder construction outputs";
  auto global_dims = input_tensor_type.global_ddim();
  auto process_mesh_attr = tensor_dist_attr.process_mesh_attr();
  auto dims_mapping = tensor_dist_attr.dims_mapping();

  auto process_mesh_shape = process_mesh_attr.shape();
  PADDLE_ENFORCE_EQ(static_cast<int>(dims_mapping.size()),
                    global_dims.size(),
                    common::errors::PreconditionNotMet(
                        "dst dims_mapping size %d does not match input size %d",
                        dims_mapping.size(),
                        global_dims.size()));

  auto local_shape = InferLocalDDim(global_dims, tensor_dist_attr);
  pir::Type out_dist_tensor_type = paddle::dialect::DistDenseTensorType::get(
      pir::IrContext::Instance(),
      input_tensor_type.dense_tensor_type(),
      tensor_dist_attr,
      local_shape);
  argument.AddOutput(out_dist_tensor_type);
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ShardTensorOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ReShardOp)
