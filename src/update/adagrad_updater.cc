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

This file is the implementation of AdaGrad updater.
*/

#include "src/update/adagrad_updater.h"

#include <cmath> // for sqrt()

namespace f2m {

#ifdef __AVX__
extern float L1_term[64];
#else
extern float L1_term[1024];
#endif

// This function need to be invoked before update.
void AdaGradUpdater::Initialize(const HyperParam& hyper_param) {
  CHECK_GT(hyper_param.learning_rate, 0);
  CHECK_GE(hyper_param.regu_lambda, 0);
  learning_rate_ = hyper_param.learning_rate;
  regu_lambda_ = hyper_param.regu_lambda;
  regu_type_ = hyper_param.regu_type;
}

// Adaptive gradient decent.
void AdaGradUpdater::Update(index_t key, real_t grad, Model* model) {
  // Do not check anything here
  std::vector<real_t>* w = model->GetParameter();
  std::vector<real_t>* cache = model->GetParamCache();
  real_t tmp = RegularTerm((*w)[key]) + grad;
  (*cache)[key] += tmp * tmp;
  (*w)[key] -= learning_rate_ * tmp * InvSqrt((*cache)[key]); // 1 / sqrt()
}

// Update model parameter in a mini-batch GD.
// Using SSE to speed up.
void AdaGradUpdater::BatchUpdate(Gradient* grad, Model* model) {
  // g /= row_len
  size_t end = model->GetLength();
  grad->Div(grad->GetMiniBatchSize());
  std::vector<real_t>* w = model->GetParameter();
  std::vector<real_t>* cache = model->GetParamCache();
  std::vector<real_t>* value = grad->GetDenseVector();
  __MX _learning_rate = _MMX_SET1_PS(learning_rate_);
  __MX _regu_lambda = _MMX_SET1_PS(regu_lambda_);
  __MX _small_num = _MMX_SET1_PS(kVerySmallNumber);
  for (size_t start_key = 0; start_key < end; start_key += _MMX_INCREMENT) {
    __MX _regular_term;
    GetRegularTerm(_regular_term, regu_type_);
    __MX _tmp = _MMX_ADD_PS(_MMX_MUL_PS(_regu_lambda, _regular_term),
                             _MMX_LOAD_PS(&(*value)[start_key]));
    __MX _cache = _MMX_LOAD_PS(&(*cache)[start_key]);
    __MX _w = _MMX_LOAD_PS(&(*w)[start_key]);
    _cache = _MMX_ADD_PS(_cache, _MMX_MUL_PS(_tmp, _tmp));
    _MMX_STORE_PS(&(*w)[start_key],
                 _MMX_SUB_PS(_w,
                 _MMX_MUL_PS(_learning_rate,
                 _MMX_MUL_PS(_tmp,
                 _MMX_RSQRT_PS(
                 _MMX_ADD_PS(_cache, _small_num))))));
    _MMX_STORE_PS(&(*cache)[start_key], _cache);
  }
}

// Update a continuous model parameter.
// Using SSE to speed up.
void AdaGradUpdater::SeqUpdate(std::vector<real_t>& value,
                               index_t start_key,
                               Model* model) {
  // Do not check anything here
  index_t end = value.size();
  std::vector<real_t>* w = model->GetParameter();
  std::vector<real_t>* cache = model->GetParamCache();
  __MX _learning_rate = _MMX_SET1_PS(learning_rate_);
  __MX _regu_lambda = _MMX_SET1_PS(regu_lambda_);
  __MX _small_num = _MMX_SET1_PS(kVerySmallNumber);
  for (index_t i = 0; i < end; i += _MMX_INCREMENT) {
    __MX _regular_term;
    GetRegularTerm(_regular_term, regu_type_);
    __MX _tmp = _MMX_ADD_PS(_MMX_MUL_PS(_regu_lambda, _regular_term),
                             _MMX_LOAD_PS(&value[i]));
    __MX _cache = _MMX_LOAD_PS(&(*cache)[start_key]);
    __MX _w = _MMX_LOAD_PS(&(*w)[start_key]);
    _cache = _MMX_ADD_PS(_cache, _MMX_MUL_PS(_tmp, _tmp));
    _MMX_STORE_PS(&(*w)[start_key],
                 _MMX_SUB_PS(_w,
                 _MMX_MUL_PS(_learning_rate,
                 _MMX_MUL_PS(_tmp,
                 _MMX_RSQRT_PS(
                 _MMX_ADD_PS(_cache, _small_num))))));
    _MMX_STORE_PS(&(*cache)[start_key], _cache);
    start_key += _MMX_INCREMENT;
  }
}

} // namespace f2m
