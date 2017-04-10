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

This file defines the Updater class that is responsible for updating
current model parameters.
*/

#ifndef F2M_UPDATE_UPDATER_H_
#define F2M_UPDATE_UPDATER_H_

#include <vector>
#include <unordered_map>
#include <x86intrin.h>

#include "src/base/common.h"
#include "src/base/math.h"
#include "src/base/class_register.h"
#include "src/data/model_parameters_in_column.h"
#include "src/data/hyper_parameters.h"

namespace f2m {

//------------------------------------------------------------------------------
// Updater function is responsible for updating current model parameters.
// The Updtaer class can be implemented by different update functions such as
// naive SGD, Momentum, Nesterov Momentum, AdaGard, RMSprop, Adam, and so on.
// On defauly, we use the naive SGD updater, which has the following form:
// [ w -= learning_rate * grad ]
//------------------------------------------------------------------------------
class Updater {
 public:
  // Constructor and Destructor.
  Updater() {  }
  virtual ~Updater() {  }

  // This function needs to be invoked before update.
  virtual void Initialize(const HyperParam& hyper_param);

  // Using naive SGD update by default.
  virtual void Update(index_t key, real_t grad, Model* model);

  // Update model parameter in a mini-batch GD
  virtual void BatchUpdate(Gradient* grad, Model* model);

  // Update a continuous model parameter.
  virtual void SeqUpdate(std::vector<real_t>& value,
                         index_t start_key,
                         Model* model);

 protected:
  // Regularizer
  real_t RegularTerm(real_t& w);

  real_t learning_rate_;
  real_t regu_lambda_;
  RegularType regu_type_; /* L1, L2 or NONE */

 private:
  DISALLOW_COPY_AND_ASSIGN(Updater);
};

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_DEFINE_REGISTRY(f2m_updater_registry, Updater);

#define REGISTER_UPDATER(format_name, updater_name)          \
  CLASS_REGISTER_OBJECT_CREATOR(                             \
      f2m_updater_registry,                                  \
      Updater,                                               \
      format_name,                                           \
      updater_name)

#define CREATE_UPDATER(format_name)                          \
  CLASS_REGISTER_CREATE_OBJECT(                              \
      f2m_updater_registry,                                  \
      format_name)

} // namespace f2m

#endif // F2M_UPDATE_UPDATER_H_

