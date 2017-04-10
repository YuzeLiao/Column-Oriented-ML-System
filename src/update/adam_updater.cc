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
Author: Yuze Liao and Chao Ma (mctt90@gmail.com)

This file is the implementation of AdamUpdater updater.
*/

#include "src/update/adam_updater.h"

#include <cmath> // for sqrt()

namespace f2m {

#ifdef __AVX__
extern float L1_term[64];
#else
extern float L1_term[1024];
#endif

// This function needs to be invoked before update.
void AdamUpdater::Initialize(const HyperParam& hyper_param) {
  CHECK_GT(hyper_param.learning_rate, 0);
  CHECK_GE(hyper_param.regu_lambda, 0);
  CHECK_GT(hyper_param.decay_rate, 0);
  CHECK_GT(hyper_param.second_decay_rate, 0);
  learning_rate_ = hyper_param.learning_rate;
  regu_lambda_ = hyper_param.regu_lambda;
  regu_type_ = hyper_param.regu_type;
  beta1_ = hyper_param.decay_rate;
  beta2_ = hyper_param.second_decay_rate;
  batch_size_ = hyper_param.batch_size;
}

// Adaptive Moment Estimation (Adam) update
void AdamUpdater::Update(index_t key, real_t grad, Model* model) {
  // Do not check anything here
  static uint64 epoch_count = 1;
  static int hit_count = 0;
  std::vector<real_t>* w = model->GetParameter();
  std::vector<real_t>* m = model->GetParamCache();
  std::vector<real_t>* v = model->GetParamCache_2();
  real_t tmp = RegularTerm((*w)[key]) + grad;
  (*m)[key] = (1-beta1_) * tmp + beta1_ * (*m)[key];
  (*v)[key] = (1-beta2_) * tmp * tmp + beta2_ * (*v)[key];
  real_t mb = (*m)[key] / (1-fastpow(beta1_, epoch_count));
  real_t vb = (*v)[key] / (1-fastpow(beta2_, epoch_count));
  (*w)[key] -= learning_rate_ * mb * InvSqrt(vb);
  hit_count++;
  if (hit_count == batch_size_) {
    epoch_count++;
    hit_count = 0;
  }
}

// Update model parameter in a mini-batch GD.
// Using SSE to speed up.
void AdamUpdater::BatchUpdate(Gradient* grad, Model* model) {
  // g /= row_len
  size_t end = model->GetLength();
  grad->Div(grad->GetMiniBatchSize());
  static uint64 epoch_count = 1;
  std::vector<real_t>* w = model->GetParameter();
  std::vector<real_t>* m = model->GetParamCache();
  std::vector<real_t>* v = model->GetParamCache_2();
  std::vector<real_t>* value = grad->GetDenseVector();
  __MX _learning_rate = _MMX_SET1_PS(learning_rate_);
  __MX _regu_lambda = _MMX_SET1_PS(regu_lambda_);
  __MX _small_num = _MMX_SET1_PS(kVerySmallNumber);
  __MX _one_minus_pow_beta1 = _MMX_SET1_PS(1-fastpow(beta1_, epoch_count));
  __MX _one_minus_pow_beta2 = _MMX_SET1_PS(1-fastpow(beta2_, epoch_count));
  __MX _beta1 = _MMX_SET1_PS(beta1_);
  __MX _beta2 = _MMX_SET1_PS(beta2_);
  for (size_t start_key = 0; start_key < end; start_key += _MMX_INCREMENT) {
   __MX _regular_term;
    GetRegularTerm(_regular_term, regu_type_);
    __MX _tmp = _MMX_ADD_PS(_MMX_MUL_PS(_regu_lambda, _regular_term),
                             _MMX_LOAD_PS(&(*value)[start_key]));
    __MX _m = _MMX_ADD_PS(_MMX_SUB_PS(_tmp,
                           _MMX_MUL_PS(_beta1, _tmp)),
                           _MMX_MUL_PS(_beta1,
                           _MMX_LOAD_PS(&(*m)[start_key])));
    _MMX_STORE_PS(&(*m)[start_key], _m);
    __MX _mb = _MMX_DIV_PS(_m, _one_minus_pow_beta1);

    _tmp = _MMX_MUL_PS(_tmp, _tmp);
    __MX _v = _MMX_ADD_PS(_MMX_SUB_PS(_tmp,
                           _MMX_MUL_PS(_beta2, _tmp)),
                           _MMX_MUL_PS(_beta2,
                           _MMX_LOAD_PS(&(*v)[start_key])));
    _MMX_STORE_PS(&(*v)[start_key], _v);
    __MX _vb = _MMX_DIV_PS(_v, _one_minus_pow_beta2);

    __MX _w = _MMX_LOAD_PS(&(*w)[start_key]);
    _MMX_STORE_PS(&(*w)[start_key],
                 _MMX_SUB_PS(_w,
                 _MMX_MUL_PS(_learning_rate,
                 _MMX_MUL_PS(_mb,
                 _MMX_RSQRT_PS(
                 _MMX_ADD_PS(_vb, _small_num))))));
  }
  epoch_count++;
}


// Update a continuous model parameter.
// Using AVX to speed up.
void AdamUpdater::SeqUpdate(std::vector<real_t>& value,
                            index_t start_key,
                            Model* model) {
  // Do not check anything here
  static uint64 epoch_count = 1;
  static int hit_count = 0;
  index_t end = value.size();
  std::vector<real_t>* w = model->GetParameter();
  std::vector<real_t>* m = model->GetParamCache();
  std::vector<real_t>* v = model->GetParamCache_2();
  __MX _learning_rate = _MMX_SET1_PS(learning_rate_);
  __MX _regu_lambda = _MMX_SET1_PS(regu_lambda_);
  __MX _small_num = _MMX_SET1_PS(kVerySmallNumber);
  __MX _one_minus_pow_beta1 = _MMX_SET1_PS(1-fastpow(beta1_, epoch_count));
  __MX _one_minus_pow_beta2 = _MMX_SET1_PS(1-fastpow(beta2_, epoch_count));
  __MX _beta1 = _MMX_SET1_PS(beta1_);
  __MX _beta2 = _MMX_SET1_PS(beta2_);
  for (index_t i = 0; i < end; i += _MMX_INCREMENT) {
    __MX _regular_term;
    GetRegularTerm(_regular_term, regu_type_);
    __MX _tmp = _MMX_ADD_PS(_MMX_MUL_PS(_regu_lambda, _regular_term),
                             _MMX_LOAD_PS(&value[i]));
    __MX _m = _MMX_ADD_PS(_MMX_SUB_PS(_tmp,
                           _MMX_MUL_PS(_beta1, _tmp)),
                           _MMX_MUL_PS(_beta1,
                           _MMX_LOAD_PS(&(*m)[start_key])));
    _MMX_STORE_PS(&(*m)[start_key], _m);
    __MX _mb = _MMX_DIV_PS(_m, _one_minus_pow_beta1);

    _tmp = _MMX_MUL_PS(_tmp, _tmp);
    __MX _v = _MMX_ADD_PS(_MMX_SUB_PS(_tmp,
                           _MMX_MUL_PS(_beta2, _tmp)),
                           _MMX_MUL_PS(_beta2,
                           _MMX_LOAD_PS(&(*v)[start_key])));
    _MMX_STORE_PS(&(*v)[start_key], _v);
    __MX _vb = _MMX_DIV_PS(_v, _one_minus_pow_beta2);

    __MX _w = _MMX_LOAD_PS(&(*w)[start_key]);
    _MMX_STORE_PS(&(*w)[start_key],
                 _MMX_SUB_PS(_w,
                 _MMX_MUL_PS(_learning_rate,
                 _MMX_MUL_PS(_mb,
                 _MMX_RSQRT_PS(
                 _MMX_ADD_PS(_vb, _small_num))))));
    start_key += _MMX_INCREMENT;
  }
  ++hit_count;
  if (hit_count == batch_size_) {
    epoch_count++;
    hit_count = 0;
  }
}

} // namespace f2m
