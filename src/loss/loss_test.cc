//------------------------------------------------------------------------------
// Copyright (c) 2016 by contributors. All Rights Reserved.
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
//------------------------------------------------------------------------------

/*
Author: Chao Ma (mctt90@gmail.com)

This file tests the Loss class.
*/

#include "gtest/gtest.h"

#include <vector>

#include "src/loss/loss.h"

namespace f2m {

class TestLoss : public Loss {
 public:
  void Predict(const DMatrix* matrix,
               Model* param,
               std::vector<real_t>& grad) {}

  void CalcGrad(const DMatrix* matrix,
                Model* param,
                Gradient* grad) {}

  real_t Evaluate(const std::vector<real_t>& pred,
                  const std::vector<real_t>& label) { return 0.0; }

  real_t CrossEntropy(const std::vector<real_t>& pred,
                      const std::vector<real_t>& label) {
    return cross_entropy_loss(pred, label);
  }

  real_t Sqaure(const std::vector<real_t>& pred,
                const std::vector<real_t>& label) {
    return square_loss(pred, label);
  }

  real_t WTX(const SparseRow* row, const std::vector<real_t>* w) {
    return wTx(row, w);
  }
};

TEST(LossTest, cross_entropy_loss) {
  TestLoss loss;
  std::vector<real_t> pred(100, 200);
  std::vector<real_t> label(100, 1.0);
  real_t objv = loss.CrossEntropy(pred, label);
  EXPECT_EQ(objv, 0.0);
}

TEST(LossTest, square_loss) {
  TestLoss loss;
  std::vector<real_t> pred(100, 10.0);
  std::vector<real_t> label(100, 8.0);
  real_t objv = loss.Sqaure(pred, label);
  EXPECT_EQ(objv, 2.0);
}

TEST(LossTest, wTx) {
  TestLoss loss;
  SparseRow row(100);
  for (int i = 0; i < 100; ++i) {
    row.X[i] = 2.0;
    row.idx[i] = i;
  }
  Model model(100, SGD);
  std::vector<real_t>* param = model.GetParameter();
  for (int i = 0; i < 100; ++i) {
    param->at(i) = 1.0;
  }
  real_t value = loss.WTX(&row, param);
  EXPECT_EQ(value, 200.0);
}

Loss* CreateLoss(const char* format_name) {
  return CREATE_LOSS(format_name);
}

TEST(LossTest, CreateLoss) {
  EXPECT_TRUE(CreateLoss("logit") != NULL);
  EXPECT_TRUE(CreateLoss("linear") != NULL);
  EXPECT_TRUE(CreateLoss("fm") != NULL);
  EXPECT_TRUE(CreateLoss("ffm") != NULL);
  EXPECT_TRUE(CreateLoss("svm") != NULL);
  EXPECT_TRUE(CreateLoss("unknow_name") == NULL);
}

} // namespace f2m
