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

This file tests svm_loss.h
*/

#include "gtest/gtest.h"

#include "src/loss/svm_loss.h"
#include "src/reader/reader.h"
#include "src/reader/parser.h"
#include "src/data/data_structure.h"
#include "src/data/model_parameters_in_column.h"

namespace f2m {

const std::string kTestfilename = "/tmp/agaricus.txt.train";
const index_t kFeatureNum = 126 + 1;
const int iteration_num = 10000;
const int kNumSamples = 100;

TEST(CalcGrad, ReadDataSetTestInmem) {
  // Parser
  Parser* parser_svm = new LibsvmParser;
  // Reader
  InmemReader reader;
  reader.Initialize(kTestfilename, kNumSamples, parser_svm, SVM);
  // Model
  Model model(kFeatureNum, SGD);
  // Gradient
  Gradient* grad = new DenseGrad;
  grad->Initialize(kFeatureNum);
  // DMatrix
  DMatrix* matrix = NULL;
  // SVMLoss
  SVMLoss loss;
  // Calc gradient
  for (int i = 0; i < iteration_num; ++i) {
    int record_num = reader.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader.GoToHead();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    loss.CalcGrad(matrix, &model, grad);
    // TODO(yuze): test the index
    grad->Reset();
    if (i % 1000 == 0) LOG(INFO) << i / 1000;
  }
}

TEST(CalcGrad, ReadDataSetTestOndisk) {
  // Parser
  Parser* parser_svm = new LibsvmParser;
  // Reader
  OndiskReader reader;
  reader.Initialize(kTestfilename, kNumSamples, parser_svm, SVM);
  // Model
  Model model(kFeatureNum, SGD);
  // Gradient
  Gradient* grad = new DenseGrad;
  grad->Initialize(kFeatureNum);
  // DMatrix
  DMatrix* matrix = NULL;
  // SVMLoss
  SVMLoss loss;
  // Calc gradient
  for (int i = 0; i < iteration_num; ++i) {
    int record_num = reader.Samples(matrix);
    if (record_num == 0) {
      --i;
      reader.GoToHead();
      continue;
    }
    EXPECT_EQ(record_num, kNumSamples);
    loss.CalcGrad(matrix, &model, grad);
    // TODO(yuze): test the index
    grad->Reset();
    if (i % 1000 == 0) LOG(INFO) << i / 1000;
  }
}

TEST(CalcGrad, ReadDataSetTestAsync) {
  // Parser
  Parser* parser_svm = new LibsvmParser;
  // Reader
  OndiskReaderAsync reader;
  reader.Initialize(kTestfilename, kNumSamples, parser_svm, SVM);
  // Model
  Model model(kFeatureNum, SGD);
  // Gradient
  Gradient* grad = new DenseGrad;
  grad->Initialize(kFeatureNum);
  // DMatrix
  DMatrix* matrix = NULL;
  // SVMLoss
  SVMLoss loss;
  // Calc gradient
  for (int i = 0; i < iteration_num; ++i) {
    int record_num = reader.Samples(matrix);
    EXPECT_EQ(record_num, kNumSamples);
    loss.CalcGrad(matrix, &model, grad);
    // TODO(yuze): test the index
    grad->Reset();
    if (i % 1000 == 0) LOG(INFO) << i / 1000;
  }
}

} // namespace f2m
