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
Copyright (c) 2016 by contributors.
Author: Chao Ma (mctt90@gmail.com)

This file defines the FMLoss class.
*/

#ifndef F2M_LOSS_FM_LOSS_H_
#define F2M_LOSS_FM_LOSS_H_

#include <x86intrin.h>
#include "src/loss/loss.h"

namespace f2m {

//------------------------------------------------------------------------------
// FMLoss is used for factorization machines task.
//------------------------------------------------------------------------------
class FMLoss : public Loss {
 public:
  FMLoss() {  }
  ~FMLoss() {  }

  // Invoke this function before we use the FMLoss class.
  void Initialize(const HyperParam& hyper_param);

  // Given the input DMatrix and current model, return the calculated gradients.
  void CalcGrad(const DMatrix* matrix,
                Model* param,
                Updater* updater);

  // Given the prediction results and the ground truth, return the loss value.
  // For factorization machines, we use the cross-entropy loss.
  real_t Evaluate(const std::vector<real_t>& pred,
                  const std::vector<real_t>& label);

 private:
  index_t max_feature_;    // The number of feature.
  int num_factor_;         // The number of latent factor.
  TaskType task_type_;     // Classification or Regression
  std::vector<real_t> tmp_result1;
  std::vector<real_t> tmp_result2;

  // over-write wTx
  void wTx(const DMatrix* matrix,
           std::vector<real_t>* w,
           std::vector<real_t>& result);

  DISALLOW_COPY_AND_ASSIGN(FMLoss);
};

} // namespace f2m

#endif // F2M_LOSS_FM_LOSS_H_
