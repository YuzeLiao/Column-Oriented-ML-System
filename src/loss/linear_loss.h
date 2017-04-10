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
Author: Chao Ma (mctt90@gmail.com)

This file defines the LinearLoss class.
*/

#ifndef F2M_LOSS_LINEAR_LOSS_H_
#define F2M_LOSS_LINEAR_LOSS_H_

#include "src/loss/loss.h"

namespace f2m {

//------------------------------------------------------------------------------
// LinearLoss is used for linear regression task.
//------------------------------------------------------------------------------
class LinearLoss : public Loss {
 public:
  LinearLoss() {  }
  ~LinearLoss() {  }

  // Given the input DMatrix and current model, return the calculated gradients.
  void CalcGrad(const DMatrix* matrix,
                Model* param,
                Updater* updater);

  // Given the prediciton results and the ground truth, return the loss value.
  // For linear regression, we use the sqaure loss.
  real_t Evaluate(const std::vector<real_t>& pred,
                  const std::vector<real_t>& label);

 private:
  DISALLOW_COPY_AND_ASSIGN(LinearLoss);
};


} // namespace f2m

#endif // F2M_LOSS_LINEAR_LOSS_H_
