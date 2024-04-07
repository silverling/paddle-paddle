// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/slogdeterminant_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/slogdeterminant_kernel_impl.h"
#include "Eigen/src/Core/Assign.h"
#include "Eigen/src/Core/AssignEvaluator.h"
#include "Eigen/src/Core/CwiseBinaryOp.h"
#include "Eigen/src/Core/CwiseNullaryOp.h"
#include "Eigen/src/Core/DenseCoeffsBase.h"
#include "Eigen/src/Core/Diagonal.h"
#include "Eigen/src/Core/GeneralProduct.h"
#include "Eigen/src/Core/GenericPacketMath.h"
#include "Eigen/src/Core/IO.h"
#include "Eigen/src/Core/NoAlias.h"
#include "Eigen/src/Core/Redux.h"
#include "Eigen/src/Core/SelfCwiseBinaryOp.h"
#include "Eigen/src/Core/TriangularMatrix.h"
#include "Eigen/src/Core/Visitor.h"
#include "Eigen/src/Core/arch/AVX/PacketMath.h"
#include "Eigen/src/Core/util/Memory.h"
#include "Eigen/src/LU/Determinant.h"
#include "Eigen/src/LU/PartialPivLU.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "src/Core/ArrayBase.h"
#include "src/Core/DenseBase.h"

PD_REGISTER_KERNEL(
    slogdet, CPU, ALL_LAYOUT, phi::SlogDeterminantKernel, float, double) {}
