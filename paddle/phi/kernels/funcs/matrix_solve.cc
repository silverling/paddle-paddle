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

#include "paddle/phi/kernels/funcs/matrix_solve.h"

#include <string>

#include "Eigen/src/Core/Assign.h"
#include "Eigen/src/Core/AssignEvaluator.h"
#include "Eigen/src/Core/CwiseBinaryOp.h"
#include "Eigen/src/Core/CwiseNullaryOp.h"
#include "Eigen/src/Core/DenseCoeffsBase.h"
#include "Eigen/src/Core/Diagonal.h"
#include "Eigen/src/Core/GeneralProduct.h"
#include "Eigen/src/Core/GenericPacketMath.h"
#include "Eigen/src/Core/NoAlias.h"
#include "Eigen/src/Core/PermutationMatrix.h"
#include "Eigen/src/Core/SelfCwiseBinaryOp.h"
#include "Eigen/src/Core/TriangularMatrix.h"
#include "Eigen/src/Core/Visitor.h"
#include "Eigen/src/Core/arch/AVX/PacketMath.h"
#include "Eigen/src/Core/util/Memory.h"
#include "src/Core/ArrayBase.h"
#include "src/Core/DenseBase.h"

namespace phi {
namespace funcs {

template <typename Context, typename T>
void MatrixSolveFunctor<Context, T>::operator()(const Context& dev_ctx,
                                                const DenseTensor& a,
                                                const DenseTensor& b,
                                                DenseTensor* out) {
  compute_solve_eigen<Context, T>(dev_ctx, a, b, out);
}

template class MatrixSolveFunctor<CPUContext, float>;
template class MatrixSolveFunctor<CPUContext, double>;

}  // namespace funcs
}  // namespace phi
