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

#include "paddle/fluid/framework/new_executor/instruction/onednn/onednn_mixed_instruction.h"

#include <functional>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dense_tensor.inl"
#include "paddle/phi/core/kernel_context.h"

namespace paddle {
namespace framework {

OneDNNMixedPhiKernelInstruction::OneDNNMixedPhiKernelInstruction(
    size_t id,
    const platform::Place& place,
    pir::Operation* op,
    const ValueExecutionInfo* value_exec_info)
    : OneDNNPhiKernelInstruction(id, place, op, value_exec_info) {
  auto op_attributes = op->attributes();
  kernel_name_ =
      op_attributes.at("kernel_name").dyn_cast<pir::StrAttribute>().AsString();
  kernel_key_ = op_attributes.at("kernel_key")
                    .dyn_cast<paddle::dialect::KernelAttribute>()
                    .data();
}

void OneDNNMixedPhiKernelInstruction::Run() {
  // Step1. Mixed Dynamic Choose Kernel
  if (!has_choose_kernel_) {
    has_choose_kernel_ = true;
    use_onednn_kernel_ =
        phi_kernel_->check_if_onednn_kernel_support_(&kernel_context_);
    if (!use_onednn_kernel_) {
      auto kernel_result =
          phi::KernelFactory::Instance().SelectKernelOrThrowError(kernel_name_,
                                                                  kernel_key_);
      delete phi_kernel_;
      phi_kernel_ = new phi::Kernel(kernel_result.kernel);
    }
  }

  // Step2. Run Kernel
  if (use_onednn_kernel_) {
    OneDNNPhiKernelInstruction::Run();
  } else {
    // TransLayout first
    auto inputs = kernel_context_.InputsBetween<phi::DenseTensor>(
        size_t(0), kernel_context_.InputsSize());

    for (size_t i = 0; i < inputs.size(); ++i) {
      auto input = inputs[i];
      if (input->layout() == phi::DataLayout::ONEDNN) {
        DataLayout tmp_layout =
            phi::OneDNNContext::tls().get_cur_paddle_data_layout();

        // NOTE(zhiqiu): to handle the special case in ApplyDataTransform() in
        // data_transfer.cc
        if (!input->IsInitialized() && tmp_layout == DataLayout::NHWC) {
          auto transed_tensor = const_cast<phi::DenseTensor*>(input);
          transed_tensor->set_layout(tmp_layout);
          phi::funcs::MatchShapeToLayout(
              transed_tensor, phi::DataLayout::ONEDNN, tmp_layout);
        } else {
          phi::DenseTensor transed_tensor;
          transed_tensor.set_meta(input->meta());
          phi::funcs::TransDataLayoutFromOneDNN(phi::DataLayout::ONEDNN,
                                                tmp_layout,
                                                *input,
                                                &transed_tensor,
                                                phi::CPUPlace());
          *(const_cast<phi::DenseTensor*>(input)) = transed_tensor;
        }
      }
    }

    VLOG(6) << "Begin run op " << phi_op_name_ << " infer meta.";
    if (infer_meta_interface_) {
      infer_meta_interface_->infer_meta_(&(infer_meta_context_));
    }
    VLOG(6) << "End run op " << phi_op_name_ << " infer meta.";
    VLOG(6) << "Begin run op " << phi_op_name_ << " kernel.";
    (*(phi_kernel_))(&(kernel_context_));
    VLOG(6) << "End run op " << phi_op_name_ << " kernel.";
  }
}

}  // namespace framework
}  // namespace paddle
