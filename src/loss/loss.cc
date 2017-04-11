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

This file is the implementation of the base Loss class.
*/

#include "src/loss/loss.h"

/* for class register */
#include "src/loss/logit_loss.h"
//#include "src/loss/linear_loss.h"
#include "src/loss/fm_loss.h"
//#include "src/loss/ffm_loss.h"
//#include "src/loss/svm_loss.h"

#include <cmath> // for log() and exp()

namespace f2m {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(f2m_loss_registry, Loss);
REGISTER_LOSS("lr", LogitLoss);
//REGISTER_LOSS("linear", LinearLoss);
REGISTER_LOSS("fm", FMLoss);
//REGISTER_LOSS("ffm", FFMLoss);
//REGISTER_LOSS("svm", SVMLoss);

// Using wTx. (different loss function can re-write wTx)
void Loss::Predict(const DMatrix* matrix,
                   Model* param,
                   std::vector<real_t>& pred) {
  CHECK_NOTNULL(matrix);
  // The pred vector should be pre-initialized.
  CHECK_GT(pred.size(), 0);
  //CHECK_EQ(pred.size(), matrix->row_len);
  std::vector<real_t>* w = param->GetParameter();
  wTx(matrix, w, pred);
}

// Cross-entropy loss.
real_t Loss::cross_entropy_loss(const std::vector<real_t>& pred,
                                const std::vector<real_t>& label) {
  real_t objv = 0.0;
  for (index_t i = 0; i < pred.size(); ++i) {
    real_t y = label[i] > 0 ? 1.0 : -1.0;
    objv += log(1.0 + fasterexp(-y*pred[i]));
  }
  return objv;
}

// Square loss.
real_t Loss::square_loss(const std::vector<real_t>& pred,
                         const std::vector<real_t>& label) {
  real_t objv = 0.0;
  for (index_t i = 0; i < pred.size(); ++i) {
    real_t tmp = label[i] - pred[i];
    objv += 0.5 * (tmp*tmp);
  }
  return objv;
}

// Hinge loss.
real_t Loss::hinge_loss(const std::vector<real_t>& pred,
                        const std::vector<real_t>& label) {
  real_t objv = 0.0;
  for (index_t i = 0; i < pred.size(); ++i) {
    real_t tmp = label[i] * pred[i];
    objv += tmp < 1 ? 1 - tmp : 0;
  }
  return objv;
}

// Calculate wTx.
void Loss::wTx(const DMatrix* matrix,
               std::vector<real_t>* w,
               std::vector<real_t>& result) {
  index_t num_y = matrix->Y[0]->size();
  memset(result.data(), 0, sizeof(real_t) * num_y);
  //printf(" result size is %lu\n", result.size());
  size_t row_len = matrix->row_len;
  // Calc real gradient
  for (size_t i = 0; i < row_len; ++i) {
    SparseRow* row = matrix->row[i];
    real_t w_i = (*w)[row->id];
    index_t col_len = row->column_len;
    for (size_t j = 0; j < col_len; ++j) {
      result[row->idx[j]] += w_i * row->X[j];
    }
  }
}

} // namespace f2m
