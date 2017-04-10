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

This file defines the AdamUpdater class.
*/

#ifndef F2M_UPDATE_ADAM_UPDATER_H_
#define F2M_UPDATE_ADAM_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/update/updater.h"

namespace f2m {

//------------------------------------------------------------------------------
// Adaptive Moment Estimation (Adam) is another method that computes adaptive
// learning rates for each parameter. In addition to sotring an exponential
// decaying average of past sqaured gradients Vt like Adadelta and RMSprop,
// Adam also keeps an exponentially decaying of past gradients Mt, similar
// to momentum. as shown in the following form:
// [ m = beta1 * m + (1 - beta1) * dx ]
// [ v = beta2 * v + (1 - beta2) * (dx ^ 2) ]
// [ w += -learning_rate * m / (sqrt(v) + 1e-7) ]
//------------------------------------------------------------------------------
class AdamUpdater : public Updater {
 public:
  // Constructor and Desstructor
  AdamUpdater() {  }
  ~AdamUpdater() {  }

  // This function needs to be invoked before update.
  void Initialize(const HyperParam& hyper_param);

  // Adaptive Moment Estimation (Adam) update
  void Update(index_t key, real_t grad, Model* model);

  // Update model parameter in a mini-batch GD
  void BatchUpdate(Gradient* grad, Model* model);

  // Update a continuous model parameter
  void SeqUpdate(std::vector<real_t>& value,
                 index_t start_key,
                 Model* model);

 private:
  real_t beta1_;
  real_t beta2_;
  int batch_size_;

  DISALLOW_COPY_AND_ASSIGN(AdamUpdater);
};

} // namespace f2m

#endif // F2M_UPDATE_ADAM_UPDATER_H_
