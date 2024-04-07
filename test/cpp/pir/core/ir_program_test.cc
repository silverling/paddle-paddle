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

#include <stdint.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "test/cpp/pir/tools/macros_utils.h"
#include "gtest/gtest-message.h"
#include "gtest/gtest-test-part.h"
#include "gtest/gtest_pred_impl.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/op_info.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/type_id.h"
#include "paddle/pir/include/core/value.h"

namespace pir {
class Builder;
}  // namespace pir

class AddOp : public pir::Op<AddOp> {
 public:
  using Op::Op;
  static const char *name() { return "test.add"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;
  void VerifySig();
  static void Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value l_operand,
                    pir::Value r_operand,
                    pir::Type sum_type);
};
void AddOp::VerifySig() {
  if (num_operands() != 2) {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "The size of inputs must be equal to 2."));
  }
  if (num_results() != 1) {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "The size of outputs must be equal to 1."));
  }
}
void AddOp::Build(pir::Builder &,
                  pir::OperationArgument &argument,
                  pir::Value l_operand,
                  pir::Value r_operand,
                  pir::Type sum_type) {
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
  argument.AddOutput(sum_type);
}
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(AddOp)
IR_DEFINE_EXPLICIT_TYPE_ID(AddOp)

TEST(program_test, slice_combine_test) {
  // (1) Init environment.
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  // (2) Create an empty program object
  pir::Program program(ctx);
  //   pir::Program *program = new pir::Program();
  EXPECT_EQ(program.block()->empty(), true);

  // (3) Create a float32 DenseTensor Parameter and save into Program
  pir::Type fp32_dtype = pir::Float32Type::get(ctx);

  // (4) Def a = ParameterOp("a")
  std::string op1_name = pir::ParameterOp::name();
  pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op1_name);
  std::unordered_map<std::string, pir::Attribute> op1_attribute{
      {"parameter_name", pir::StrAttribute::get(ctx, "a")}};
  pir::Operation *op1 =
      pir::Operation::Create({}, op1_attribute, {fp32_dtype}, op1_info);
  program.block()->push_back(op1);

  // (5) Def b = Constant("b")
  std::string op2_name = std::string(pir::ConstantOp::name());
  pir::OpInfo op2_info = ctx->GetRegisteredOpInfo(op2_name);
  pir::AttributeMap attr_map;
  attr_map.insert(std::pair<std::string, pir::Attribute>(
      "value", pir::FloatAttribute::get(ctx, 2.0)));
  pir::Operation *op2 =
      pir::Operation::Create({}, attr_map, {fp32_dtype}, op2_info);
  program.block()->push_back(op2);

  // (6) Def combine_op = CombineOp("a", "b")
  std::string combine_op_name = std::string(pir::CombineOp::name());
  pir::OpInfo combine_op_info = ctx->GetRegisteredOpInfo(combine_op_name);
  pir::Type output_type = pir::VectorType::get(
      ctx, std::vector<pir::Type>({fp32_dtype, fp32_dtype}));
  pir::Operation *combine_op = pir::Operation::Create(
      {op1->result(0), op2->result(0)}, {}, {output_type}, combine_op_info);
  pir::CombineOp combine_op_type = combine_op->dyn_cast<pir::CombineOp>();
  EXPECT_TRUE(combine_op_type.out());
  program.block()->push_back(combine_op);

  // (7) Def slice_op = SliceOp(combine_op, 0)
  std::string slice_op_name = std::string(pir::SliceOp::name());
  pir::OpInfo slice_op_info = ctx->GetRegisteredOpInfo(slice_op_name);
  pir::Attribute index_attr = pir::Int32Attribute::get(ctx, 0);
  pir::Operation *slice_op = pir::Operation::Create({combine_op->result(0)},
                                                    {{"index", index_attr}},
                                                    {fp32_dtype},
                                                    slice_op_info);
  program.block()->push_back(slice_op);

  // (8) Traverse Program
  EXPECT_EQ(program.block()->size() == 4, true);
}
