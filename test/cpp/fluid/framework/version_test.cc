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

#include "paddle/fluid/framework/version.h"

#include "gtest/gtest-message.h"
#include "gtest/gtest-test-part.h"
#include "gtest/gtest_pred_impl.h"

namespace paddle {
namespace framework {
TEST(Version, Basic) {
  EXPECT_TRUE(IsProgramVersionSupported(0));
  EXPECT_TRUE(IsProgramVersionSupported(1));
  EXPECT_TRUE(IsProgramVersionSupported(-1));

  EXPECT_TRUE(IsTensorVersionSupported(0));
  EXPECT_TRUE(IsTensorVersionSupported(1));
  EXPECT_TRUE(IsTensorVersionSupported(-1));
}
}  // namespace framework
}  // namespace paddle
