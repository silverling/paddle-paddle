// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/memory_optimize_pass/recurrent_op_eager_deletion_pass.h"

#include <unordered_map>
#include <vector>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/operators/controlflow/op_variant.h"
#include "paddle/utils/any.h"

namespace paddle {
namespace framework {
namespace ir {

using paddle::operators::OpAndGradOpPair;
using paddle::operators::OpVariantSet;

void RecurrentOpEagerDeletionPass::ApplyImpl(Graph *graph) const {
  // Find all recurrent_op and recurrent_grad_op in graph
  // Note the graph only contains ops and block 0
  std::unordered_map<size_t, OpAndGradOpPair> target_ops =
      DeviceIdToRecurrentAndRecurrentGradOp(*graph);

  if (graph->IsConstructedByPartialProgram()) {
    PADDLE_ENFORCE_LE(target_ops.size(),
                      1,
                      platform::errors::InvalidArgument(
                          "Unsupported multi devices if graph is constructed "
                          "with partial program."));
    size_t scope_idx = 0;
    auto &recur_ops = target_ops[scope_idx].first;
    auto &recur_grad_ops = target_ops[scope_idx].second;

    auto all_ops = graph->OriginProgram().Block(0).AllOps();
    if (recur_ops.empty()) {
      operators::AppendOpVariantByOpName(
          all_ops, std::string("recurrent"), &recur_ops);
    } else if (recur_grad_ops.empty()) {
      operators::AppendOpVariantByOpName(
          all_ops, std::string("recurrent_grad"), &recur_grad_ops);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "One of recur_ops or recur_grad_ops should be empty."));
    }
  }

  for (auto &entry : target_ops) {
    // Prepare safe eager deletion on different devices because the garbage
    // collection may be different across devices
    OpAndGradOpPair &op_pair = entry.second;
    PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(
        graph->OriginProgram(), &op_pair);
  }

  auto all_ops = ir::FilterByNodeWrapper<details::OpHandleBase>(*graph);
  for (auto op_hander : all_ops) {
    auto *compute_op = dynamic_cast<details::ComputationOpHandle *>(op_hander);
    if (compute_op == nullptr) continue;
    if (compute_op->Name() == "recurrent" ||
        compute_op->Name() == "recurrent_grad") {
      ir::Node *op_node = op_hander->Node();
      auto *op_base = compute_op->GetOp();
      if (op_base->Attrs().count("skip_eager_deletion_vars")) {
        op_node->Op()->SetAttr("skip_eager_deletion_vars",
                               op_base->Attrs().at("skip_eager_deletion_vars"));
      }
    }
  }
}

// Returns a std::unordered_map mapping from the device id to recurrent op and
// grad op pair
std::unordered_map<size_t, OpAndGradOpPair>
RecurrentOpEagerDeletionPass::DeviceIdToRecurrentAndRecurrentGradOp(
    const Graph &graph) const {
  std::unordered_map<size_t, OpAndGradOpPair> ret;
  std::vector<details::OpHandleBase *> all_ops =
      FilterByNodeWrapper<details::OpHandleBase>(graph);

  for (auto *op : all_ops) {
    auto compute_op = dynamic_cast<details::ComputationOpHandle *>(op);
    if (compute_op == nullptr) continue;

    if (compute_op->Name() == "recurrent") {
      // GetScopeIdx() returns device/place id
      ret[compute_op->GetScopeIdx()].first.emplace(compute_op->GetOp());
    } else if (compute_op->Name() == "recurrent_grad") {
      // GetScopeIdx() returns device/place id
      ret[compute_op->GetScopeIdx()].second.emplace(compute_op->GetOp());
    }
  }
  return ret;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(recurrent_op_eager_deletion_pass,
              paddle::framework::ir::RecurrentOpEagerDeletionPass);
