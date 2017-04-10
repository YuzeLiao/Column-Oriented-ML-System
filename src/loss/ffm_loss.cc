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

This file is the implementation of FFM.
*/

#include "src/loss/ffm_loss.h"

namespace f2m {

// Get some hyper parameters.
void FFMLoss::Initialize(const HyperParam& hyper_param) {
  CHECK_GT(hyper_param.max_feature, 0);
  CHECK_GT(hyper_param.num_factor, 0);
  CHECK_GT(hyper_param.num_field, 0);
  max_feature_ = hyper_param.max_feature;
  num_factor_ = hyper_param.num_factor;
  num_field_ = hyper_param.num_field;
  is_sparse_ = hyper_param.is_sparse;
  if (hyper_param.is_train && !is_sparse_) {
    grad_ = new Gradient;
    grad_->Initialize(hyper_param.num_param);
  }
  matrix_size_ = num_field_ * num_factor_;
  task_type_ = hyper_param.task_type;
}

// Math: [ partial_grad * X ] for linear term
//       [ partial_grad * X_j * X_k * w_k_fj ] for index j
//       [ partial_grad * X_j * X_k * w_j_fk ] for index k
// partial_grad refers to [ -y / (1 + (1 / exp(-y * <w,x>))) ]
void FFMLoss::CalcGrad(const DMatrix* matrix,
                       Model* param,
                       Updater* updater) {
  CHECK_NOTNULL(matrix);
  CHECK_GT(matrix->row_len, 0);
  CHECK_NOTNULL(updater);
  static std::vector<real_t> k_vec_j(num_factor_);
  static std::vector<real_t> k_vec_k(num_factor_);
  real_t* p_j = k_vec_j.data();
  real_t* p_k = k_vec_k.data();
  std::vector<real_t> *w = param->GetParameter();
  size_t row_len = matrix->row_len;
  // Calc gradient
  for (size_t i = 0; i < row_len; ++i) {
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
    const real_t* array_ = w->data();
    for (index_t j = 1; j < col_len; ++j) {
      real_t x_j_mul_pg = row->X[j] * pg;
      index_t j_mul_mxs_add_mf = row->idx[j] * matrix_size_ + max_feature_;
      index_t j_mul_fac_add_mf = row->field[j] * num_factor_ + max_feature_;
      for (index_t k = j+1; k < col_len; ++k) {
        index_t pos_j =
            row->field[k] * num_factor_ + j_mul_mxs_add_mf;
        index_t pos_k =
            row->idx[k] * matrix_size_ + j_mul_fac_add_mf;
        real_t j_mul_k_mul_pg = x_j_mul_pg * row->X[k];
        __MX val_tmp = _MMX_SET1_PS(j_mul_k_mul_pg);
        for (index_t p = 0; p < num_factor_; p += _MMX_INCREMENT) {
          __MX wj = _MMX_MUL_PS(val_tmp, _MMX_LOAD_PS(array_ + pos_k + p));
          __MX wk = _MMX_MUL_PS(val_tmp, _MMX_LOAD_PS(array_ + pos_j + p));
          _MMX_STORE_PS(p_j+p, wj);
          _MMX_STORE_PS(p_k+p, wk);
        }
        // update K
        if (is_sparse_) {
          updater->SeqUpdate(k_vec_j, pos_j, param);
          updater->SeqUpdate(k_vec_k, pos_k, param);
        } else {
          grad_->SeqAddgrad(k_vec_j, pos_j);
          grad_->SeqAddgrad(k_vec_k, pos_k);
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

// Return cross-entropy loss.
real_t FFMLoss::Evaluate(const std::vector<real_t>& pred,
                         const std::vector<real_t>& label) {
  CHECK_GT(pred.size(), 0);
  CHECK_GT(label.size(), 0);
  if (task_type_ == Regression) {
    return this->square_loss(pred, label);
  } else {
    return this->cross_entropy_loss(pred, label);
  }
}

// Math: [ pred = <w,x> + <v_i_fj, v_j_fi> * x_i * x_j ]
real_t FFMLoss::wTx(const SparseRow* row, const std::vector<real_t>* w) {
  real_t val = 0.0;
  index_t col_len = row->column_len;
  // linear term
  for (index_t i = 0; i < col_len; ++i) {
    index_t pos = row->idx[i];
    val += (*w)[pos] * row->X[i];
  }
  // latent factor
  const float* array_ = &((*w)[0]);
  __MX accumulate_ = _MMX_SETZERO_PS();
  for (index_t i = 1; i < col_len; ++i) {
    index_t idx_i_mul_mxs_add_mf = row->idx[i] * matrix_size_ + max_feature_;
    index_t field_i_mul_fac_mf = row->field[i] * num_factor_ + max_feature_;
    real_t x_i = row->X[i];
    for (index_t j = i+1; j < col_len; ++j) {
      index_t pos_i = idx_i_mul_mxs_add_mf + row->field[j] * num_factor_;
      index_t pos_j = row->idx[j] * matrix_size_ + field_i_mul_fac_mf;
      real_t x_i_mul_x_j = row->X[j] * x_i;
      __MX x_i_x_j = _MMX_SET1_PS(x_i_mul_x_j);
      for (index_t k = 0; k < num_factor_; k += _MMX_INCREMENT) {
        __MX wi = _MMX_LOAD_PS(array_ + pos_i + k);
        __MX wj = _MMX_LOAD_PS(array_ + pos_j + k);
        accumulate_ = _MMX_ADD_PS(accumulate_,
                                  _MMX_MUL_PS(wi,
                                  _MMX_MUL_PS(wj, x_i_x_j)));
      }
    }
  }
#ifdef __AVX__
  accumulate_ = _MMX_HADD_PS(accumulate_, accumulate_);
  real_t tmp[8];
  _MMX_STORE_SS(tmp, accumulate_);
  val = tmp[0];
#else
  _MMX_STORE_SS(&val, accumulate_);
#endif

  return val;
}

} // namespace f2m
