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

This file defines the Loss class, which is the core part of machine
learning algorithms.
*/

#ifndef F2M_LOSS_LOSS_H_
#define F2M_LOSS_LOSS_H_

#include <vector>
#include <cmath>  // for exp() and log()

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/data/data_structure.h"
#include "src/data/model_parameters_in_column.h"
#include "src/data/hyper_parameters.h"
#include "src/update/updater.h"

namespace f2m {

//------------------------------------------------------------------------------
// The Loss is an abstract class, which can be implemented by the real
// loss functions such as logistic regression loss (logit_loss.h), FM loss
// (fm_loss.h), linear regression loss (linear_loss.h), svm loss (svm_loss.h),
// as well as FFM loss (ffm_loss.h).
//------------------------------------------------------------------------------
class Loss {
 public:
  // Constructor and Destructor.
  Loss() {  }
  virtual ~Loss() {  }

  // Invoke this function before we use the Loss class.
  virtual void Initialize(const HyperParam& hyper_param) {
    is_sparse_ = hyper_param.is_sparse;
    if (hyper_param.is_train && !is_sparse_) {
      grad_ = new Gradient;
      grad_->Initialize(hyper_param.num_param);
    }
  }

  // Given the input DMatrix and current model, return the prediction results.
  virtual void Predict(const DMatrix* matrix,
                       Model* param,
                       std::vector<real_t>& pred);

  // Given the input DMatrix and current model, return the calculated gradients.
  virtual void CalcGrad(const DMatrix* matrix,
                        Model* param,
                        Updater* updater) = 0;

  // Given the prediction results and the groudtruth, return the loss value.
  virtual real_t Evaluate(const std::vector<real_t>& pred,
                          const std::vector<real_t>& label) = 0;

 protected:
  // Define the cross-entropy loss.
  // Note that the cross-entropy loss takes -1 and 1 for positive and
  // negative examples, respectivly.
  real_t cross_entropy_loss(const std::vector<real_t>& pred,
                            const std::vector<real_t>& label);

  // Define the square loss.
  real_t square_loss(const std::vector<real_t>& pred,
                     const std::vector<real_t>& label);

  // Define the hinge loss
  real_t hinge_loss(const std::vector<real_t>& pred,
                    const std::vector<real_t>& label);

  // Calculate wTx.
  virtual void wTx(const DMatrix* matrix,
                     std::unordered_map<index_t, real_t>* w,
                     std::vector<real_t>& result);

  std::vector<real_t> result;
  Gradient* grad_;   // Storing gradient in dense model
  bool is_sparse_;   // Dense or sparse

 private:
  DISALLOW_COPY_AND_ASSIGN(Loss);
};

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_DEFINE_REGISTRY(f2m_loss_registry, Loss);

#define REGISTER_LOSS(format_name, loss_name)          \
  CLASS_REGISTER_OBJECT_CREATOR(                       \
      f2m_loss_registry,                               \
      Loss,                                            \
      format_name,                                     \
      loss_name)

#define CREATE_LOSS(format_name)                       \
  CLASS_REGISTER_CREATE_OBJECT(                        \
      f2m_loss_registry,                               \
      format_name)

} // namespace f2m

#endif // F2M_LOSS_LOSS_H_
