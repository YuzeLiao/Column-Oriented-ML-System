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

This file is the implementation of Model and Gradient class.
*/

#include <pmmintrin.h> // for SSE

#include "src/data/model_parameters_in_column.h"

#include "src/base/common.h"
#include "src/base/file_util.h"
#include "src/base/math.h"
#include "src/base/stringprintf.h"

namespace f2m {

//------------------------------------------------------------------------------
// The Model class
//------------------------------------------------------------------------------

// Parameters for gaussian distribution.
static const real_t kInitMean = 0.0;
static const real_t kInitStdev = 0.01;

// Basic contributor.
Model::Model(size_t parameter_num, UpdaterType type, bool gaussian) :
  parameters_num_(parameter_num), updater_type_(type) {
  CHECK_GE(parameters_num_, 0);
  try {
    parameters_.resize(parameters_num_, 0.0);
    if (updater_type_ == AdaGrad || updater_type_ == Momentum
        || updater_type_ == RMSprop) {
      param_cache_.resize(parameters_num_, 0.0);
    } else if (updater_type_ == AdaDelta || updater_type_ == Adam) {
      param_cache_.resize(parameters_num_, 0.0);
      param_cache_2_.resize(parameters_num_, 0.0);
    }
    if (gaussian) {
      InitModelUsingGaussian();
    }
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for current      \
                   model parameters. Parameter size: "
               << parameters_num_;
  }
}

// Initialize model from a checkpoint file.
Model::Model(const std::string& filename, UpdaterType type) {
  CHECK_NE(filename.empty(), true);
  updater_type_ = type;
  this->LoadModel(filename);
}

// Serialize model to a checkpoint file.
void Model::SaveModel(const std::string& filename) {
  CHECK_NE(filename.empty(), true);
  FILE* file_ptr_param =
      OpenFileOrDie(StringPrintf("%s_param", filename.c_str()).c_str(), "w");
  // Write param
  WriteVectorToFile<real_t>(file_ptr_param, this->parameters_);
  Close(file_ptr_param);
  // Write param_cache
  if (updater_type_ == AdaGrad || updater_type_ == Momentum
      || updater_type_ == RMSprop) {
    FILE* file_ptr_param_cache =
        OpenFileOrDie(StringPrintf("%s_cache", filename.c_str()).c_str(), "w");
    WriteVectorToFile<real_t>(file_ptr_param_cache, this->param_cache_);
    Close(file_ptr_param_cache);
  } else if (updater_type_ == AdaDelta || updater_type_ == Adam) {
    FILE* file_ptr_param_cache =
        OpenFileOrDie(StringPrintf("%s_cache", filename.c_str()).c_str(), "w");
    FILE* file_ptr_param_cache_2 =
        OpenFileOrDie(StringPrintf("%s_cache_2", filename.c_str()).c_str(), "w");
    WriteVectorToFile<real_t>(file_ptr_param_cache, this->param_cache_);
    WriteVectorToFile<real_t>(file_ptr_param_cache_2, this->param_cache_2_);
    Close(file_ptr_param_cache_2);
  }
}

// Deserialize model from a checkpoint file.
void Model::LoadModel(const std::string& filename) {
  CHECK_NE(filename.empty(), true);
  FILE* file_ptr_param =
      OpenFileOrDie(StringPrintf("%s_param", filename.c_str()).c_str(), "r");
  // Load param
  ReadVectorFromFile<real_t>(file_ptr_param, this->parameters_);
  parameters_num_ = parameters_.size();
  Close(file_ptr_param);
  // Load param_cache
  if (updater_type_ == AdaGrad || updater_type_ == Momentum
      || updater_type_ == RMSprop) {
    FILE* file_ptr_param_cache =
        OpenFileOrDie(StringPrintf("%s_cache", filename.c_str()).c_str(), "r");
    ReadVectorFromFile<real_t>(file_ptr_param_cache, this->param_cache_);
    Close(file_ptr_param_cache);
  } else if (updater_type_ == AdaDelta || updater_type_ == Adam) {
    FILE* file_ptr_param_cache =
        OpenFileOrDie(StringPrintf("%s_cache", filename.c_str()).c_str(), "r");
    FILE* file_ptr_param_cache_2 =
        OpenFileOrDie(StringPrintf("%s_cache_2", filename.c_str()).c_str(), "r");
    ReadVectorFromFile<real_t>(file_ptr_param_cache, this->param_cache_);
    ReadVectorFromFile<real_t>(file_ptr_param_cache_2, this->param_cache_2_);
    Close(file_ptr_param_cache);
    Close(file_ptr_param_cache_2);
  }
}

// Reset current model to init state.
void Model::Reset(bool gaussian) {
  if (gaussian) {
    InitModelUsingGaussian();
  } else {
    for (size_t i = 0; i < parameters_num_; ++i) {
      parameters_[i] = 0.0;
    }
  }
}

// Save model parameters to a tmp vector
void Model::Saveweight(std::vector<real_t>& vec) {
  CHECK_EQ(parameters_num_, vec.size());
  copy(parameters_.begin(), parameters_.end(), vec.begin());
}

// Load model parameters from a temp vector
void Model::Loadweight(const std::vector<real_t>& vec) {
  CHECK_EQ(parameters_num_, vec.size());
  copy(vec.begin(), vec.end(), parameters_.begin());
}

// Initialize model parameters using Gaussian distribution.
void Model::InitModelUsingGaussian() {
  CHECK_EQ(parameters_num_, parameters_.size());
  for (size_t i = 0; i < parameters_num_; ++i) {
    parameters_[i] = ran_gaussion(kInitMean, kInitStdev);
  }
}

// Delete the model file and cache file.
void Model::RemoveModelFile(const std::string filename) {
  // Remove model file
  RemoveFile(StringPrintf("%s_param", filename.c_str()).c_str());
  if (updater_type_ == AdaGrad || updater_type_ == Momentum
      || updater_type_ == RMSprop) {
    RemoveFile(StringPrintf("%s_cache", filename.c_str()).c_str());
  } else if (updater_type_ == AdaDelta || updater_type_ == Adam) {
    RemoveFile(StringPrintf("%s_cache", filename.c_str()).c_str());
    RemoveFile(StringPrintf("%s_cache_2", filename.c_str()).c_str());
  }
}

//------------------------------------------------------------------------------
// The Gradient class
//------------------------------------------------------------------------------

// Initialize gradient vector
void Gradient::Initialize(size_t num_parameters) {
  CHECK_GT(num_parameters, 0);
  num_param_ = num_parameters;
  try {
    grad_.resize(num_param_, 0.0);
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for gradient vector."
               << "Number of parameter: " << num_param_;
  }
}

// Batch add gradient and use SSE to speed up
void Gradient::SeqAddgrad(std::vector<real_t>& value, index_t start_key) {
  real_t* v = value.data();
  real_t* g = grad_.data();
  for (index_t i = 0; i < value.size(); i += _MMX_INCREMENT) {
    /* Same as: grad_[start_key] += value[i];
                ++start_key;                  */
    __MX _value = _MMX_LOAD_PS(v+i);
    __MX _grad = _MMX_LOAD_PS(g+start_key);
    _MMX_STORE_PS(g+start_key, _MMX_ADD_PS(_grad, _value));
    start_key += _MMX_INCREMENT;
  }
}

// All elements are divided by a number
// Using SSE to speed up
void Gradient::Div(real_t value) {
  real_t* g = grad_.data();
  __MX _value = _MMX_SET1_PS(value);
  for (size_t i = 0; i < grad_.size(); i += _MMX_INCREMENT) {
    /* same as: grad_[i] /= value; */
    __MX _grad = _MMX_LOAD_PS(g+i);
    _MMX_STORE_PS(g+i, _MMX_DIV_PS(_grad, _value));
  }
}

// Reset current gradient vector
void Gradient::Reset() {
  for (size_t i = 0; i < num_param_; ++i) {
     grad_[i] = 0.0;
  }
}



} // namespace f2m
