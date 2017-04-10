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

This file declares flags used by f2m to accept instructions. For
descriptions on these flags, please refer to flags.cc. For more details,
refer to the design document.
*/

#ifndef F2M_TRAIN_FLAGS_H_
#define F2M_TRAIN_FLAGS_H_

#include <string>

#include "gflags/gflags.h"

#include "src/reader/reader.h"
#include "src/reader/parser.h"
#include "src/data/data_structure.h"
#include "src/data/model_parameters_in_column.h"
#include "src/data/hyper_parameters.h"
#include "src/loss/loss.h"
#include "src/update/updater.h"

namespace f2m {

//------------------------------------------------------------------------------
// Most flags should be accessed using the following accessors.
//------------------------------------------------------------------------------
DECLARE_bool(f2m_is_train);
DECLARE_string(f2m_task_type);
DECLARE_string(f2m_model_type);
DECLARE_bool(f2m_is_sparse);
DECLARE_double(f2m_learning_rate);
DECLARE_string(f2m_file_format);
DECLARE_int32(f2m_num_factor);
DECLARE_string(f2m_updater);
DECLARE_double(f2m_decay_rate);
DECLARE_double(f2m_second_decay_rate);
DECLARE_double(f2m_regu_lambda);
DECLARE_string(f2m_regu_type);
DECLARE_string(f2m_train_set_file);
DECLARE_string(f2m_test_set_file);
DECLARE_string(f2m_model_checkpoint_file);
DECLARE_int32(f2m_num_iteration);
DECLARE_bool(f2m_cross_validation);
DECLARE_int32(f2m_num_folds);
DECLARE_bool(f2m_in_memory_trainning);
DECLARE_int32(f2m_batch_size);
DECLARE_bool(f2m_early_stop);
DECLARE_bool(f2m_sigmoid);
DECLARE_string(f2m_log_filebase);

//-----------------------------------------------------------------------------
// Check the correctness of flags.
//-----------------------------------------------------------------------------
bool ValidateCommandLineFlags();

//-----------------------------------------------------------------------------
// Set the hyper param
//-----------------------------------------------------------------------------
void SetHyperParam(HyperParam& hyper_param);

//-----------------------------------------------------------------------------
// Get the log filename
//-----------------------------------------------------------------------------
std::string LogFilebase();

//-----------------------------------------------------------------------------
// Invoke ValidateCommandLineFlags() before using the following accessors.
//-----------------------------------------------------------------------------
Reader* CreateReader();
Parser* CreateParser();
Loss* CreateLoss();
Updater* CreateUpdater();

bool IfGaussian();

} // namespace f2m

#endif // F2M_TRAIN_FLAGS_H_
