/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/cholesky_kernel.h"

#include <algorithm>
#include <new>
#include <string>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "Eigen/src/Cholesky/LLT.h"
#include "Eigen/src/Core/Assign.h"
#include "Eigen/src/Core/AssignEvaluator.h"
#include "Eigen/src/Core/CwiseBinaryOp.h"
#include "Eigen/src/Core/CwiseNullaryOp.h"
#include "Eigen/src/Core/DenseCoeffsBase.h"
#include "Eigen/src/Core/Dot.h"
#include "Eigen/src/Core/GeneralProduct.h"
#include "Eigen/src/Core/GenericPacketMath.h"
#include "Eigen/src/Core/Map.h"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/NoAlias.h"
#include "Eigen/src/Core/Redux.h"
#include "Eigen/src/Core/SelfAdjointView.h"
#include "Eigen/src/Core/SelfCwiseBinaryOp.h"
#include "Eigen/src/Core/SolveTriangular.h"
#include "Eigen/src/Core/Transpose.h"
#include "Eigen/src/Core/TriangularMatrix.h"
#include "Eigen/src/Core/arch/AVX/PacketMath.h"
#include "Eigen/src/Core/arch/SSE/PacketMath.h"
#include "Eigen/src/Core/products/SelfadjointProduct.h"
#include "Eigen/src/Core/util/Constants.h"
#include "Eigen/src/Core/util/Memory.h"
#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/dense_tensor.h"
#include "src/Core/ArrayBase.h"
#include "src/Core/DenseBase.h"

namespace phi {

template <typename T, typename Context>
void CholeskyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    bool upper,
                    DenseTensor* out) {
  using EigenMatrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using InputMatrixMap = Eigen::Map<const EigenMatrix>;
  using OutputMatrixMap = Eigen::Map<EigenMatrix>;

  auto& dims = x.dims();
  int batch_count = 1;
  for (int i = 0; i < dims.size() - 2; i++) {
    batch_count *= static_cast<int>(dims[i]);
  }
  auto m = dims[dims.size() - 1];

  const auto* x_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);
  // Cholesky decomposition for each matrix, maybe can use multi threads
  for (int i = 0; i < batch_count; i++) {
    auto input = InputMatrixMap(x_data + i * m * m, m, m);
    auto output = OutputMatrixMap(out_data + i * m * m, m, m);
    if (upper) {
      Eigen::LLT<
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
          Eigen::UpLoType::Upper>
          llt_decomposition(input);
      PADDLE_ENFORCE_EQ(llt_decomposition.info(),
                        Eigen::Success,
                        errors::InvalidArgument(
                            "Cholesky decomposition was not successful. The "
                            "%d-th input matrice "
                            "might not be not be positive definite.",
                            i));
      output = llt_decomposition.matrixU();
    } else {
      Eigen::LLT<
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
          Eigen::UpLoType::Lower>
          llt_decomposition(input);
      PADDLE_ENFORCE_EQ(llt_decomposition.info(),
                        Eigen::Success,
                        errors::InvalidArgument(
                            "Cholesky decomposition was not successful. The "
                            "%d-th input matrice "
                            "might not be not be positive definite.",
                            i));
      output = llt_decomposition.matrixL();
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    cholesky, CPU, ALL_LAYOUT, phi::CholeskyKernel, float, double) {}
