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

This file defines the AdaGradUpdater class.
*/

#ifndef F2M_UPDATE_ADAGRAD_UPDATER_H_
#define F2M_UPDATE_ADAGRAD_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/update/updater.h"

namespace f2m {

//------------------------------------------------------------------------------
// AdaGrad is an algorithm for gradient-based optimization that does just
// this: It adapts the learning rate to the parameters, performing larger
// updates for infrequent and smaller updates for frequent parameters. For
// this reason, it is well-suited for dealing with sparse data. AdaGrad uses
// the following update:
// [ cache += dx ^ 2 ]
// [ w += -learning_rate * dx / (sqrt(cache) + 1e-7) ]
//------------------------------------------------------------------------------
class AdaGradUpdater : public Updater {
 public:
  // Constructor and Desstructor
  AdaGradUpdater() {  }
  ~AdaGradUpdater() {  }

  // This function neede to be invoked before update.
  void Initialize(const HyperParam& hyper_param);

  // AdaGrad Update.
  void Update(index_t key, real_t grad, Model* model);

  // Update model parameter in a mini-batch GD
  void BatchUpdate(Gradient* grad, Model* model);

  // Update a continuous model parameter
  void SeqUpdate(std::vector<real_t>& value,
                 index_t start_key,
                 Model* model);

 private:
  DISALLOW_COPY_AND_ASSIGN(AdaGradUpdater);
};

} // namespace f2m

#endif // F2M_UPDATE_ADAGRAD_UPDATER_H_
