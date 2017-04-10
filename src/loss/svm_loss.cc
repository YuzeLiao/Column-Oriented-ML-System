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
Author: Chao Ma (mctt90@gmail.com) and Yuze Liao.

This file is the implementation of Support Vector Machine.
*/

#include "src/loss/svm_loss.h"

namespace f2m {

void SVMLoss::CalcGrad(const DMatrix* matrix,
                       Model* param,
                       Updater* updater) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_len, 0);
  CHECK_NOTNULL(updater);
  std::vector<real_t> *w = param->GetParameter();
  size_t row_len = matrix->row_len;
  // Calc gradient
  for (size_t i = 0; i < row_len; ++i) {
    SparseRow* row = matrix->row[i];
    index_t col_len = row->column_len;
    real_t pred = wTx(row, w);
    real_t y = matrix->Y[i] > 0 ? 1.0 : -1.0;
    for (index_t j = 0; j < col_len; ++j) {
      real_t w_j = (y * pred) < 1 ? -y * row->X[j] : 0;
      if (is_sparse_) {
        updater->Update(row->idx[j], w_j, param);
      } else {
        grad_->Addgrad(row->idx[j], w_j);
      }
    }
  }
  // Updating in dense model
  if (!is_sparse_) {
    grad_->SetMiniBatchSize(row_len);
    updater->BatchUpdate(grad_, param);
    grad_->Reset();
  }
}

real_t SVMLoss::Evaluate(const std::vector<real_t>& pred,
                         const std::vector<real_t>& label) {
  CHECK_GT(pred.size(), 0);
  CHECK_GT(label.size(), 0);
  return this->hinge_loss(pred, label);
}

} // namespace f2m
