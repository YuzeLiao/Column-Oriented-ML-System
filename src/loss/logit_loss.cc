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

This file is the implementation of logistic regression.
*/

#include "src/loss/logit_loss.h"

namespace f2m {

// Get some hyper parameter
void LogitLoss::Initialize(const HyperParam& hyper_param) {
  is_sparse_ = hyper_param.is_sparse;
  if (hyper_param.is_train) {
    grad_ = new Gradient;
    grad_->Initialize(hyper_param.max_feature);
  }
  result.resize(hyper_param.batch_size, 0);
}

// Math: [ (-y / ((1/exp(-y*<w,x>)) + 1)) * X]
void LogitLoss::CalcGrad(const DMatrix* matrix,
                         Model* param,
                         Updater* updater) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_len, 0);
  CHECK_NOTNULL(updater);
  std::vector<real_t> *w = param->GetParameter();
  size_t row_len = matrix->row_len;
  // Calc real gradient
  index_t num_y = matrix->Y[0]->size();
  wTx(matrix, w, result);
  for (size_t i = 0; i < num_y; ++i) {
    real_t y = (*matrix->Y[0])[i] > 0 ? 1.0 : -1.0;
    result[i] = -y / (1.0 + (1.0 / fasterexp(-y * result[i])));
  }
  real_t realGrad = 0.0;
  for (size_t i = 0; i < row_len; ++i) {
    SparseRow* row = matrix->row[i];
    index_t col_len = row->column_len;
    for (size_t j = 0; j < col_len; ++j) {
      realGrad += result[row->idx[j]] * row->X[j];
    }
    realGrad /= num_y;
    grad_->Addgrad(row->id, realGrad);
  }
  // Updating in dense model
  updater->BatchUpdate(grad_, param);
  grad_->Reset();
}

// Return cross-entropy loss.
real_t LogitLoss::Evaluate(const std::vector<real_t>& pred,
                           const std::vector<real_t>& label) {
  CHECK_GT(pred.size(), 0);
  CHECK_GT(label.size(), 0);
  return this->cross_entropy_loss(pred, label);
}

} // namespace f2mmZ
