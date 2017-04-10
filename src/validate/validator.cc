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

This file is the implementation of the Validator.
*/

#include <vector>
#include <iostream>

#include "src/validate/validator.h"
#include "src/data/data_structure.h"

namespace f2m {

real_t Validator::Validate(Model* model, Reader* rd_train, Reader* rd_val, int iter) {
  if (iter != -1) {
    std::cout << "iteration: " << iter << "  ";
  }
  real_t train_loss = validate(model, rd_train);
  std::cout << "train loss: " << train_loss << "  ";

  if (rd_val != NULL) {
    real_t val_loss = validate(model, rd_val);
    std::cout << "validation loss: " << val_loss << std::endl;
    return val_loss;
  }
  std::cout << std::endl;

  return train_loss;
}

// Give current model and data, return evaluated loss.
real_t Validator::validate(Model* model, Reader* reader) {
  reader->GoToHead();
  DMatrix* matrix = nullptr;
  std::vector<real_t> pred;
  real_t loss_val = 0.0;
  uint64 total_size = 0;
  // Read until end of file
  while (reader->Samples(matrix)) {
    if (matrix->Y[0]->size() != pred.size()) {
      pred.resize(matrix->Y[0]->size());
    }
    loss_->Predict(matrix, model, pred);
    loss_val += loss_->Evaluate(pred, (*matrix->Y[0]));
    total_size += matrix->Y[0]->size();
  }
  loss_val /= total_size;
  reader->GoToHead();

  return loss_val;
}

} // namespace f2m
