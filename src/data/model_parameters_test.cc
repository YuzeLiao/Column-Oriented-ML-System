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

This file tests model_parameters.h
*/

#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "src/data/model_parameters_in_column.h"
#include "src/base/file_util.h"

namespace f2m {

const uint32 kParameter_num = 2500 * 4; // Assume kParameter_num % 4 == 0
const std::string kFilename = "/tmp/test_model.binary";

//------------------------------------------------------------------------------
// Model test
//------------------------------------------------------------------------------

TEST(MODEL_TEST, Init) {
  // Init model using gaussion.
  Model model_lr(kParameter_num, SGD, true);
  std::vector<real_t>* para = model_lr.GetParameter();
  EXPECT_EQ(para->size(), kParameter_num);
}

TEST(MODEL_TEST, SaveModel) {
  // Init model (set all parameters to zero)
  Model model_lr(kParameter_num, Adam);
  model_lr.SaveModel(kFilename);
}

TEST(MODEL_TEST, LoadModel) {
  // Init model with gaussion distribution.
  Model model_lr(kParameter_num, Adam, true);
  // parameters become 0
  model_lr.LoadModel(kFilename);
  std::vector<real_t>* para = model_lr.GetParameter();
  for (index_t i = 0; i < para->size(); ++i) {
    EXPECT_EQ((*para)[i], (real_t)0.0);
  }
  std::vector<real_t>* cache_1 = model_lr.GetParamCache();
  for (index_t i = 0; i < cache_1->size(); ++i) {
    EXPECT_EQ((*cache_1)[i], (real_t)0.0);
  }
  std::vector<real_t>* cache_2 = model_lr.GetParamCache_2();
  for (index_t i = 0; i < cache_2->size(); ++i) {
    EXPECT_EQ((*cache_2)[i], (real_t)0.0);
  }
}

TEST(MODEL_TEST, InitModelFromDiskfile) {
  Model model_lr(kFilename, Adam);
  std::vector<real_t>* para = model_lr.GetParameter();
  for (index_t i = 0; i < para->size(); ++i) {
    EXPECT_EQ((*para)[i], (real_t)0.0);
  }
}

TEST(MODEL_TEST, RemoveFile) {
  Model model_lr(kParameter_num, Adam, true);
  model_lr.RemoveModelFile(kFilename.c_str());
}

TEST(MODEL_TEST, SaveweightAndLoadweight) {
  Model model_lr(kParameter_num, Adam);
  std::vector<real_t> vec(kParameter_num, 1.0);
  model_lr.Saveweight(vec);
  for (index_t i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], 0);
    vec[i] = 2.0;
  }
  model_lr.Loadweight(vec);
  std::vector<real_t>* para = model_lr.GetParameter();
  for (index_t i = 0; i < para->size(); ++i) {
    EXPECT_EQ((*para)[i], 2.0);
  }
}

//------------------------------------------------------------------------------
// Gradient test
//------------------------------------------------------------------------------

TEST(GRADIENT_TEST, TestGradient) {
  Gradient grad;
  grad.Initialize(kParameter_num);
  EXPECT_EQ(grad.GetLength(), kParameter_num);
  std::vector<real_t>* grad_ptr = grad.GetDenseVector();
  for (int i = 0; i < kParameter_num; ++i) {
    grad.Addgrad(i, 4.0);
  }
  grad.Div(4.0);
  for (int i = 0; i < kParameter_num; ++i) {
    EXPECT_EQ((*grad_ptr)[i], 1.0);
  }
}

} // namespace f2m
