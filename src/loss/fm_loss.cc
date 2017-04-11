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
  if (hyper_param.is_train) {
    grad_ = new Gradient;
    grad_->Initialize(hyper_param.num_param);
  }
  task_type_ = hyper_param.task_type;
  result.resize(hyper_param.batch_size, 0);
  tmp_result1.resize(hyper_param.batch_size, 0);
  tmp_result2.resize(hyper_param.batch_size, 0);
}

// Return cross-entropy loss.
real_t FMLoss::Evaluate(const std::vector<real_t>& pred,
                        const std::vector<real_t>& label) {
  CHECK_GT(pred.size(), 0);
  CHECK_GT(label.size(), 0);
  return this->cross_entropy_loss(pred, label);
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
  index_t num_y = matrix->Y[0]->size();
  wTx(matrix, w, result);
  for (size_t i = 0; i < num_y; ++i) {
      real_t y = (*matrix->Y[0])[i] > 0 ? 1.0 : -1.0;
      result[i] = -y / (1.0 + (1.0 /fasterexp(-y * result[i])));
  }
  for (size_t i = 0; i < row_len; ++i) {
    SparseRow* row = matrix->row[i];
    index_t col_len = row->column_len;
    real_t realGrad = 0.0;
    for (size_t j = 0; j < col_len; ++j) {
      realGrad += result[row->idx[j]] * row->X[j];
    }
    realGrad /= num_y;
    grad_->Addgrad(row->id, realGrad);
  }

  for (size_t i = 1; i <= num_factor_; ++i) {
    size_t bias = i * max_feature_;
    for (size_t j = 1; j < row_len; ++j) {
      SparseRow* row = matrix->row[j];
      index_t col_len = row->column_len;
      real_t w_i = (*w)[row->id + bias];
      for (size_t k = 0; k < col_len; ++k) {
        tmp_result2[row->idx[k]] +=  row->X[k] * w_i;
      }
    }
    for (size_t j = 1; j < row_len; ++j) {
      SparseRow* row = matrix->row[j];
      index_t pos = row->id + bias;
      index_t col_len = row->column_len;
      real_t realGrad = 0.0;
      real_t w_i = (*w)[pos];
      for (size_t k = 0; k < col_len; ++k) {
        real_t x = row->X[k];
        index_t idx = row->idx[k];
        realGrad += result[idx] * (tmp_result2[idx] - w_i * x) * x;
      }
      realGrad /= num_y;
      grad_->Addgrad(pos, realGrad);
    }
    memset(tmp_result2.data(), 0, sizeof(real_t) * num_y);
  }
  updater->BatchUpdate(grad_, param);
  grad_->Reset();
}


void FMLoss::wTx(const DMatrix* matrix,
               std::vector<real_t>* w,
               std::vector<real_t>& result) {
  index_t num_y = matrix->Y[0]->size();
  memset(result.data(), 0, sizeof(real_t) * num_y);
  memset(tmp_result1.data(), 0, sizeof(real_t) * num_y);
  memset(tmp_result2.data(), 0, sizeof(real_t) * num_y);
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
 
  for (size_t i = 1; i <= num_factor_; ++i) {
    size_t bias = i * max_feature_;
    for (size_t j = 1; j < row_len; ++j) {
      SparseRow* row = matrix->row[j];
      real_t w_i = (*w)[row->id + bias];
     // printf("|%lu| ", row->id + bias);
      index_t col_len = row->column_len;
      for (size_t k = 0; k < col_len; ++k) {
        real_t vx = row->X[k] * w_i;
        tmp_result1[row->idx[k]] -= vx * vx;
        tmp_result2[row->idx[k]] += vx;  
      }
    }
    for (size_t k = 0; k < num_y; ++k) {
      tmp_result1[k] += tmp_result2[k] * tmp_result2[k];
      tmp_result2[k] = 0;
    }
  }
  for (size_t i = 0; i < num_y; ++i) {
    result[i] += 0.5 * tmp_result1[i];
  }
  //printf("\n\n\n");
}

} // namespace f2m
