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

This file defines the SVMLoss class.
*/

#ifndef F2M_LOSS_SVM_LOSS_H_
#define F2M_LOSS_SVM_LOSS_H_

#include "src/loss/loss.h"

namespace f2m {

//------------------------------------------------------------------------------
// SVMLoss is used for svm task.
//------------------------------------------------------------------------------
class SVMLoss : public Loss {
 public:
  SVMLoss() {  }
  ~SVMLoss() {  }

  // Given the input DMatrix and current model, return the calculated gradients.
  void CalcGrad(const DMatrix* matrix,
                Model* param,
                Updater* updater);

  // Given the prediction results and the groud truth, return the loss value.
  // For SVM, we use the hinge loss.
  real_t Evaluate(const std::vector<real_t>& pred,
                  const std::vector<real_t>& label);

 private:
  DISALLOW_COPY_AND_ASSIGN(SVMLoss);
};

} // namespace f2m

#endif // F2M_LOSS_SVM_LOSS_H_
