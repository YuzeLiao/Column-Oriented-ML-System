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

This file defines the Validator that is responsible for validating
current model and controlling the trainning process.
*/

#ifndef F2M_VALIDATE_VALIDATOR_H_
#define F2M_VALIDATE_VALIDATOR_H_

#include "src/base/common.h"
#include "src/data/model_parameters_in_column.h"
#include "src/loss/loss.h"
#include "src/reader/reader.h"

namespace f2m {

//------------------------------------------------------------------------------
// Validator is responsible for validating current model.
//------------------------------------------------------------------------------
class Validator {
 public:
  Validator() { }
  ~Validator() { }

  // Invoke this function before we use this class.
  void Initialize(Loss* loss) {
    CHECK_NOTNULL(loss);
    loss_ = loss;
  }

  // Given current model and data, return current loss.
  real_t Validate(Model* model, Reader* rd_train, Reader* rd_val = NULL, int iter = -1);

 protected:
  Loss* loss_;
  int  batch_size_;

  real_t validate(Model* model, Reader* reader);

 private:
  DISALLOW_COPY_AND_ASSIGN(Validator);
};

} // namespace f2m

#endif // F2M_VALIDATE_VALIDATOR_H_
