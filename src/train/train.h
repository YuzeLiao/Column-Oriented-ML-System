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

This file defines the control process in f2m.
*/

#ifndef F2M_TRAIN_TRAIN_H_
#define F2M_TRAIN_TRAIN_H_

#include <vector>

#include "src/reader/reader.h"

namespace f2m {

//------------------------------------------------------------------------------
// Initialization and finalization of f2m.
//------------------------------------------------------------------------------
bool Initialize();

void Finalize();

void ReadProblem(Reader* reader, index_t* max_feature, int* num_field);

//------------------------------------------------------------------------------
// Train model
//------------------------------------------------------------------------------
void StartTrainWork();

//------------------------------------------------------------------------------
// Train without Cross-validation
//------------------------------------------------------------------------------
void Train(const std::vector<Reader*>& reader_list);

//------------------------------------------------------------------------------
// Train with cross-validation
//------------------------------------------------------------------------------
void CVTrain(const std::vector<Reader*>& reader_list, int train_num);

//------------------------------------------------------------------------------
// Start predict work
//------------------------------------------------------------------------------
void StartPredictWork();

void WritePredToFile(const std::vector<real_t>& pred, FILE* file);

// Given the original prediction result,
// return the new result transformed by the sigmoid function.
void SigmoidTrans(std::vector<real_t>& pred);

} // namespace f2m

#endif // F2M_TRAIN_TRAIN_H_
