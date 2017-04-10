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

This file defines the FFMLoss class.
*/

#ifndef F2M_LOSS_FFM_LOSS_H_
#define F2M_LOSS_FFM_LOSS_H_

#include <pmmintrin.h>

#include "src/loss/loss.h"

namespace f2m {

//------------------------------------------------------------------------------
// FFMLoss is used for field-aware factorization machines tasks.
//------------------------------------------------------------------------------
class FFMLoss : public Loss {
 public:
  FFMLoss() {  }
  ~FFMLoss() {  }

  // Invoke this function before we use the FFMLoss class.
  void Initialize(const HyperParam& hyper_param);

  // Given the input DMatrix and current model, return the calculated gradients.
  void CalcGrad(const DMatrix* matrix,
                Model* param,
                Updater* updater);

  // Given the prediction results and the groud truth, return the loss value.
  // For FFM we use the cross-entropy loss.
  real_t Evaluate(const std::vector<real_t>& pred,
                  const std::vector<real_t>& label);

 private:
  index_t max_feature_;    // The number of feature.
  int num_field_;          // The number of field.
  int num_factor_;         // The number of latent factor.
  int matrix_size_;        // num_field_ * num_factor_
  TaskType task_type_;     // Regression or Classification

  // over-write wTx
  real_t wTx(const SparseRow* row, const std::vector<real_t>* w);

  DISALLOW_COPY_AND_ASSIGN(FFMLoss);
};

} // namespace f2m

#endif // F2M_LOSS_FFM_LOSS_H_
