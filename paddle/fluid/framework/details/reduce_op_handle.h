//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <ostream>
#include <unordered_map>
#include <utility>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace phi {
class SelectedRows;
}  // namespace phi

namespace paddle {
namespace framework {

namespace details {
struct VarHandle;
}  // namespace details
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
namespace platform {
class NCCLContextMap;
}  // namespace platform
}  // namespace paddle
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#elif defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#endif

namespace paddle {
namespace framework {
class Scope;

namespace details {
struct CollectiveContext {
  std::vector<std::string> endpoints_;
  int trainer_id_{0};

  std::string String() const {
    std::stringstream ss;
    ss << "endpoints_:";
    for (auto e : endpoints_) {
      ss << e << ",";
    }

    ss << "trainer_id_:" << trainer_id_;

    return ss.str();
  }

  static CollectiveContext *GetInstance() {
    std::call_once(init_flag_,
                   [&]() { context_.reset(new CollectiveContext()); });
    return context_.get();
  }

 private:
  static std::once_flag init_flag_;
  static std::unique_ptr<CollectiveContext> context_;
};

struct ReduceOpHandle : public OpHandleBase {
  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  const platform::NCCLContextMap *nccl_ctxs_;
  ReduceOpHandle(ir::Node *node,
                 const std::vector<Scope *> &local_scopes,
                 const std::vector<platform::Place> &places,
                 const platform::NCCLContextMap *nccl_ctxs)
      : OpHandleBase(node),
        local_scopes_(local_scopes),
        places_(places),
        nccl_ctxs_(nccl_ctxs) {
    if (nccl_ctxs_) {
      for (auto &p_ctx : nccl_ctxs_->contexts_) {
        this->SetDeviceContext(platform::CUDAPlace(p_ctx.first),
                               p_ctx.second.ctx_.get());
      }
    }
  }
#elif defined(PADDLE_WITH_XPU_BKCL)
  const platform::BKCLContextMap *bkcl_ctxs_;
  ReduceOpHandle(ir::Node *node,
                 const std::vector<Scope *> &local_scopes,
                 const std::vector<platform::Place> &places,
                 const platform::BKCLContextMap *bkcl_ctxs)
      : OpHandleBase(node),
        local_scopes_(local_scopes),
        places_(places),
        bkcl_ctxs_(bkcl_ctxs) {
    if (bkcl_ctxs_) {
      for (auto &p_ctx : bkcl_ctxs_->contexts_) {
        this->SetDeviceContext(platform::XPUPlace(p_ctx.first),
                               p_ctx.second.ctx_.get());
      }
    }
  }
#else
  ReduceOpHandle(ir::Node *node,
                 const std::vector<Scope *> &local_scopes,
                 const std::vector<platform::Place> &places)
      : OpHandleBase(node), local_scopes_(local_scopes), places_(places) {}
#endif

  std::string Name() const override;

  bool IsMultiDeviceTransfer() override { return true; };

 protected:
  void RunImpl() override;

  std::vector<Scope *> GetLocalScopes() override { return local_scopes_; }

#if (defined PADDLE_WITH_CUDA || defined PADDLE_WITH_HIP) && \
    defined PADDLE_WITH_DISTRIBUTE
  template <typename DevCtx, typename DataType>
  void GatherSelectedRows(
      const std::vector<const phi::SelectedRows *> &src_selecte_rows_,
      const std::vector<platform::Place> &in_places,
      const std::map<platform::Place, platform::DeviceContext *> &dev_ctxes,
      VarHandle *out_var_handle,
      const platform::Place &out_place,
      phi::SelectedRows *dst_selecte_rows);
#endif

  void Wait(
      const std::map<platform::Place, platform::DeviceContext *> &dev_ctxes);

  template <typename T>
  std::vector<const T *> GetInputValues(
      const std::vector<VarHandle *> &in_var_handles,
      const std::vector<Scope *> &var_scopes) const;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
