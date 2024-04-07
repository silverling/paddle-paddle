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

#include <vector>

#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/program.h"
#include "test/cpp/pir/tools/test_dialect.h"
#include "test/cpp/pir/tools/test_op.h"
#include "gtest/gtest-message.h"
#include "gtest/gtest-test-part.h"
#include "gtest/gtest_pred_impl.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/block_operand.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/iterator.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/value.h"

TEST(block_operand_test, type_block) {
  pir::IrContext ctx;
  ctx.GetOrRegisterDialect<test::TestDialect>();

  pir::Program program(&ctx);
  pir::Block* block = program.block();

  pir::Builder builder(&ctx, block);
  test::RegionOp region_op = builder.Build<test::RegionOp>();

  auto& region = region_op->region(0);

  pir::Block* block_1 = new pir::Block();
  pir::Block* block_2 = new pir::Block();
  pir::Block* block_3 = new pir::Block();
  region.push_back(block_1);
  region.push_back(block_2);
  region.push_back(block_3);

  builder.SetInsertionPointToBlockEnd(block_1);
  auto op1 = builder.Build<test::BranchOp>(std::vector<pir::Value>{}, block_2);
  EXPECT_TRUE(block_2->HasOneUse());
  EXPECT_FALSE(block_2->use_empty());

  auto iter_begin = block_2->use_begin();
  auto iter_end = block_2->use_end();
  auto block_operand = op1->block_operand(0);
  auto iter_curr = iter_begin++;
  EXPECT_EQ(iter_begin, iter_end);
  EXPECT_EQ(*iter_curr, block_operand);
  EXPECT_EQ(block_2->first_use(), block_operand);
  EXPECT_EQ(iter_curr->owner(), op1);

  builder.SetInsertionPointToBlockEnd(block_3);
  auto op3 = builder.Build<test::BranchOp>(std::vector<pir::Value>{}, block_1);
  block_operand = op3->block_operand(0);
  block_operand.set_source(block_2);
  EXPECT_EQ(block_2, block_operand.source());
}
