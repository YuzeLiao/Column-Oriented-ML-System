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

This file is the implementation of factorization machines.
*/

#include "src/loss/fm_loss.h"

namespace f2m {

// Get some hyper parameters.
void FMLoss::Initialize(const HyperParam& hyper_param) {
  CHECK_GT(hyper_param.max_feature, 0);
  CHECK_GT(hyper_param.num_factor, 0);
  max_feature_ = hyper_param.max_feature;
  num_factor_ = hyper_param.num_factor;
  is_sparse_ = hyper_param.is_sparse;
  if (hyper_param.is_train && !is_sparse_) {
    grad_ = new Gradient;
    grad_->Initialize(hyper_param.num_param);
  }
  task_type_ = hyper_param.task_type;
}

// Return cross-entropy loss.
real_t FMLoss::Evaluate(const std::vector<real_t>& pred,
                        const std::vector<real_t>& label) {
  CHECK_GT(pred.size(), 0);
  CHECK_GT(label.size(), 0);
  if (task_type_ == Regression) {
    return this->square_loss(pred, label);
  } else {
    return this->cross_entropy_loss(pred, label);
  }
}

// Math: [ partial_grad * X ] for linear term
//       [ partial_grad * X_j * X_k * w_k ] for index j
//       [ partial_grad * X_j * X_k * w_j ] for index k
// partial_grad refers to [ -y / (1 + 1/exp(-y*<w,x>)) ]
void FMLoss::CalcGrad(const DMatrix* matrix,
                      Model* param,
                      Updater* updater) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_len, 0);
  CHECK_NOTNULL(updater);
  std::vector<real_t> *w = param->GetParameter();
  size_t row_len = matrix->row_len;
  // Calc real gradient
  for (index_t i = 0; i < row_len; ++i) {
    SparseRow* row = matrix->row[i];
    index_t col_len = row->column_len;
    real_t pred = wTx(row, w);
    real_t y = 0, pg = 0;
    if (task_type_ == Regression) {
      y = matrix->Y[i];
      pg = pred - y;
    } else {
      y = matrix->Y[i] > 0 ? 1.0 : -1.0;
      pg = -y / (1.0 + (1.0 / fasterexp(-y * pred)));
    }
    // for linear term
    for (index_t j = 0; j < col_len; ++j) {
      real_t realGrad = pg * row->X[j];
      if (is_sparse_) {
        updater->Update(row->idx[j], realGrad, param);
      } else {
        grad_->Addgrad(row->idx[j], realGrad);
      }
    }
    // for latent factor
    for (index_t k = 0; k < num_factor_; ++k) {
      real_t v_mul_x = 0.0;
      index_t fac_add_k = max_feature_ + k;
      for (index_t j = 1; j < col_len; ++j) {
        index_t pos = row->idx[j] * num_factor_ + fac_add_k;
        real_t v = (*w)[pos];
        real_t x = row->X[j];
        v_mul_x += (x*v);
      }
      for (index_t j = 1; j < col_len; ++j) {
        index_t pos = row->idx[j] * num_factor_ + fac_add_k;
        real_t x = row->X[j];
        real_t v = (*w)[pos];
        real_t realGrad = (x*v_mul_x - v*x*x) * pg;
        if (is_sparse_) {
          updater->Update(pos, realGrad, param);
        } else {
          grad_->Addgrad(pos, realGrad);
        }
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

// Math: [ pred = <w,x> + <v_i, v_j> * x_i * x_j ]
// Here we use a mathmatic trick to reduce the complexity
// from O(kn*n) to O(k*n)
real_t FMLoss::wTx(const SparseRow* row, const std::vector<real_t>* w) {
  real_t val =  0.0;
  index_t col_len = row->column_len;
  // linear term
  for (index_t i = 0; i < col_len; ++i) {
    index_t pos = row->idx[i];
    val += (*w)[pos] * row->X[i];
  }
  // latent factor
  real_t tmp = 0.0;
  for (index_t k = 0; k < num_factor_; ++k) {
    real_t square_sum = 0.0, sum_sqaure = 0.0;
    for (index_t i = 0; i < col_len; ++i) {
      real_t x = row->X[i];
      index_t pos = row->idx[i] * num_factor_ + max_feature_ + k;
      real_t v = (*w)[pos];
      square_sum += (x*v);
      sum_sqaure += (x*x*v*v);
    }
    square_sum *= square_sum;
    tmp += (square_sum - sum_sqaure);
  }
  val += (0.5 * tmp);

  return val;
}

} // namespace f2m
