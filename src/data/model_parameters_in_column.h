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

This file defines the model parameters and gradient used by f2m.
*/

#ifndef F2M_DATA_MODEL_PARAMETERS_H_
#define F2M_DATA_MODEL_PARAMETERS_H_

#include <vector>
#include <string>
#include <unordered_map>

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/data/data_structure.h"

namespace f2m {

//------------------------------------------------------------------------------
// The Model class is responsible for storing global model prameters, which
// will be represented in a flat way, that is, no matter what Ml model we
// use, such as LR, FM, or FFM, we store the model parameters in a big array.
// We can make a checkpoint for current model, and we can also load a model
// checkpoint from target disk file.
//------------------------------------------------------------------------------
class Model {
 public:
  // Default Constructor and Destructor
  Model() { }
  ~Model() { }
  // Set all parameters to 0 or using Gaussian distribution.
  explicit Model(size_t parameter_num, UpdaterType type, bool gaussian = false);

  // Initialize model parameters from a checkpoint file.
  explicit Model(const std::string& filename, UpdaterType type);

  // Serialize model to a checkpoint file.
  void SaveModel(const std::string& filename);

  // Deserialize model from a checkpoint file.
  void LoadModel(const std::string& filename);

  // Get the pointer of current model parameters.
  inline std::vector<real_t>* GetParameter() { return &parameters_; }

  // Get the pointer of current model cache_1.
  inline std::vector<real_t>* GetParamCache() { return &param_cache_; }

  // Get the pointer of current model cache_2.
  inline std::vector<real_t>* GetParamCache_2() { return &param_cache_2_; }

  // Get the length of current model parameters.
  inline index_t GetLength() { return parameters_num_; }

  // Reset current model to init state.
  void Reset(bool gaussion = false);

  // Save model parameters to a temp vector.
  void Saveweight(std::vector<real_t>& vec);

  // Load model parameters from a temp vector.
  void Loadweight(const std::vector<real_t>& vec);

  // Delete the model file and cahce file.
  void RemoveModelFile(const std::string filename);

 protected:
  std::vector<real_t> parameters_;       // Storing the model parameters.
  std::vector<real_t> param_cache_;      // Cache_1 for some parameter update functions.
  std::vector<real_t> param_cache_2_;    // Cache_2 for some parameter update functions.
  size_t              parameters_num_;   // Number of model parameters.
  UpdaterType         updater_type_;     // What updater we use in this task.

  // Initialize model using Gaussian distribution.
  void InitModelUsingGaussian();

 private:
  DISALLOW_COPY_AND_ASSIGN(Model);
};

//------------------------------------------------------------------------------
// The Gradient is used to store the calculated gradients
// during computation.
//------------------------------------------------------------------------------
class Gradient {
 public:
  Gradient() {  }
  virtual ~Gradient() {  }

  // Initialize gradient vector
  void Initialize(size_t num_parameters);

  // Add temp gradient during computation
  inline void Addgrad(index_t key, real_t value) {
    //printf("key is %u\n", key);
    grad_[key] += value;
  }

  // Batch add gradient and use SSE to speed up
  //void SeqAddgrad(std::vector<real_t>& value, index_t start_key);

  // Get one element of gradient.
  inline real_t Getgrad(index_t key) {
    return grad_[key];
  }

  // We need to div mini-batch size in model update
  inline void SetMiniBatchSize(size_t size) {
    CHECK_GT(size, 0);
    batch_size_ = size;
  }

  // Return mini-batch size.
  inline size_t GetMiniBatchSize() {
    return batch_size_;
  }

  // All the gradients are divied by a value
  // Using SSE to speed up
  //void Div(real_t value);

  // Reset current gradient vector
  void Reset();

  // Get the pointer of current dense vector
  inline std::unordered_map<index_t, real_t>* GetDenseVector() {
    return &grad_;
  }

  // Return the length of model parameters
  inline index_t GetLength() { return num_param_; }

 private:
  size_t batch_size_;         // Mini-batch size
  std::unordered_map<index_t, real_t> grad_;  // To store dense data
  size_t num_param_;          // Number of model parameters

  DISALLOW_COPY_AND_ASSIGN(Gradient);
};

} // namespace f2m

#endif // F2M_DATA_MODEL_PARAMETERS_H_
