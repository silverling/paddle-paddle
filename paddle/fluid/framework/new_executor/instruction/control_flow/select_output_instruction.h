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
#include <string>
#include <vector>

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/platform/place.h"

namespace pir {
class Operation;
}  // namespace pir

namespace paddle {
namespace framework {
class ValueExecutionInfo;
class Variable;

class SelectOutputInstruction : public InstructionBase {
 public:
  SelectOutputInstruction(size_t id,
                          const platform::Place& place,
                          ::pir::Operation* op,
                          ValueExecutionInfo* value_exe_info);

  void Run() override;

  const std::string& Name() const override { return name_; }

  ::pir::Operation* Operation() const override { return op_; }

 private:
  ::pir::Operation* op_;

  OpFuncType type_;

  std::string name_{"pd_op.select_output"};

  Variable* mask_;  // not owned

  Variable* input_;  // not owned

  std::vector<Variable*> outputs_;  // not owned
};

}  // namespace framework
}  // namespace paddle
