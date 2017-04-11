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

This file is the implementation of Updater.
*/

/* for class register */
#include "src/update/regular_term.h"
#include "src/update/updater.h"
//#include "src/update/adam_updater.h"
//#include "src/update/adagrad_updater.h"
//#include "src/update/adadelta_updater.h"
//#include "src/update/momentum_updater.h"
//#include "src/update/rmsprop_updater.h"

namespace f2m {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(f2m_updater_registry, Updater);
REGISTER_UPDATER("sgd", Updater);
//REGISTER_UPDATER("adam", AdamUpdater);
//REGISTER_UPDATER("adagrad", AdaGradUpdater);
//REGISTER_UPDATER("adadelta", AdaDeltaUpdater);
//REGISTER_UPDATER("momentum", MomentumUpdater);
//REGISTER_UPDATER("rmsprop", RMSPropUpdater);

// User need to invoke this function before updating.
void Updater::Initialize(const HyperParam& hyper_param) {
  CHECK_GT(hyper_param.learning_rate, 0);
  CHECK_GE(hyper_param.regu_lambda, 0);
  learning_rate_ = hyper_param.learning_rate;
  regu_lambda_ = hyper_param.regu_lambda;
  regu_type_ = hyper_param.regu_type;
}

// Naive SGD updater.
void Updater::Update(index_t key, real_t grad, Model* model) {
  // Do not check anything here
  //printf("key is %u\n", key);
  std::vector<real_t>* w = model->GetParameter();
  // w -= eta * g
  (*w)[key] -= learning_rate_ * (RegularTerm((*w)[key]) + grad);
}

// Update model parameter in a mini-batch GD.
// Using SSE to speed up.
void Updater::BatchUpdate(Gradient* grad, Model* model) {
  // g /= row_len
  //size_t end = model->GetLength();
  //grad->Div(grad->GetMiniBatchSize());
  std::vector<real_t>* w = model->GetParameter();
  std::unordered_map<index_t, real_t>* grad_ = grad->GetDenseVector();
  std::unordered_map<index_t, real_t>::iterator it = grad_->begin();
  std::unordered_map<index_t, real_t>::iterator end = grad_->end();
  while(it != end) {
    (*w)[it->first] -= learning_rate_ * (RegularTerm((*w)[it->first]) + it->second);
    ++it;
  }
  
  
  /*std::vector<real_t>* value = grad->GetDenseVector();
  __MX _learning_rate = _MMX_SET1_PS(learning_rate_);
  __MX _regu_lambda = _MMX_SET1_PS(regu_lambda_);
  for (size_t start_key = 0; start_key < end; start_key += _MMX_INCREMENT) {
    __MX _regular_term;
    GetRegularTerm(_regular_term, regu_type_);
    __MX _grad = _MMX_LOAD_PS(&(*value)[start_key]);
    __MX _w = _MMX_LOAD_PS(&(*w)[start_key]);
    __MX _delta_g = _MMX_MUL_PS(_learning_rate,
                                 _MMX_ADD_PS(_grad,
                                 _MMX_MUL_PS(_regular_term, _regu_lambda)));
    _MMX_STORE_PS(&(*w)[start_key], _MMX_SUB_PS(_w, _delta_g));
  }*/
}

// Update a continuous model parameter.
// Using SSE to speed up.
void Updater::SeqUpdate(std::vector<real_t>& value,
                        index_t start_key,
                        Model* model) {
  //static float * regu = new float(value.size());
  index_t end = value.size();
  std::vector<real_t>* w = model->GetParameter();
  __MX _learning_rate = _MMX_SET1_PS(learning_rate_);
  __MX _regu_lambda = _MMX_SET1_PS(regu_lambda_);
  for (index_t i = 0; i < end; i += _MMX_INCREMENT) {
    __MX _regular_term;
    GetRegularTerm(_regular_term, regu_type_);
    __MX _grad = _MMX_LOAD_PS(&value[i]);
    __MX _w = _MMX_LOAD_PS(&(*w)[start_key]);
    __MX _delta_g = _MMX_MUL_PS(_learning_rate,
                                 _MMX_ADD_PS(_grad,
                                 _MMX_MUL_PS(_regular_term, _regu_lambda)));
    _MMX_STORE_PS(&(*w)[start_key], _MMX_SUB_PS(_w, _delta_g));
    start_key += _MMX_INCREMENT;
  }
}

// Regularizer
real_t Updater::RegularTerm(real_t& w) {
  switch (regu_type_) {
    case L2: return regu_lambda_ * w; break;
    case L1: return regu_lambda_ * (w > 0 ? 1 : -1); break;
    default: return 0.0;
  }
  LOG(FATAL) << "Error: Unknow regularizer.";
}

} // namespace f2m
