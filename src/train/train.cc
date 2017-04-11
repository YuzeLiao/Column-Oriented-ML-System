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

This file is the implementation of the train.h file.
*/

#include <string>
#include <set>

#include "src/train/train.h"

#include "src/base/common.h"
#include "src/base/file_util.h"
#include "src/base/scoped_ptr.h"
#include "src/base/stringprintf.h"
#include "src/base/math.h"
#include "src/data/data_structure.h"
#include "src/data/hyper_parameters.h"
#include "src/data/model_parameters_in_column.h"
#include "src/reader/reader.h"
#include "src/reader/file_splitor.h"
#include "src/reader/parser.h"
#include "src/loss/loss.h"
#include "src/update/updater.h"
#include "src/validate/validator.h"
#include "src/train/flags.h"

namespace f2m {

using std::vector;
using std::string;

//------------------------------------------------------------------------------
// F2M context, using poor guy's singleton.
//------------------------------------------------------------------------------

scoped_ptr<FileSpliter>& GetFileSpliter() {
  static scoped_ptr<FileSpliter> spliter;
  return spliter;
}

scoped_ptr<Parser>& GetParser() {
  static scoped_ptr<Parser> parser;
  return parser;
}

scoped_ptr<Loss>& GetLoss() {
  static scoped_ptr<Loss> loss;
  return loss;
}

scoped_ptr<Updater>& GetUpdater() {
  static scoped_ptr<Updater> updater;
  return updater;
}

scoped_ptr<Validator>& GetValidator() {
  static scoped_ptr<Validator> validator;
  return validator;
}

scoped_ptr<HyperParam>& GetHyperParam() {
  static scoped_ptr<HyperParam> hyper_param;
  return hyper_param;
}

scoped_ptr<Model>& GetModel() {
  static scoped_ptr<Model> model;
  return model;
}

//------------------------------------------------------------------------------
// Initialization and Finalization of f2m.
//------------------------------------------------------------------------------

bool Initialize() {
  bool bo = true;

  // Redirect log message to disk files from terminal. Note that
  // LogFilebase() is valid only after ValidateCommandLineFlags.
  std::string filename_prefix = LogFilebase();
  InitializeLogger(StringPrintf("%s.INFO", filename_prefix.c_str()),
  	               StringPrintf("%s.WARN", filename_prefix.c_str()),
  	               StringPrintf("%s.ERROR", filename_prefix.c_str()));

  LOG(PRINT) << "Start to initialize the context of f2m.";

  // Create the HyperParam
  GetHyperParam().reset(new HyperParam);
  if (GetHyperParam().get() == nullptr) {
  	LOG(ERROR) << "Create HyperParam error.";
  	bo = false;
  }
  SetHyperParam((*GetHyperParam().get()));

  LOG(PRINT) << "Initialize hyper parameters successfully.";

  // Create the Parser
  GetParser().reset(CreateParser());
  if (GetParser().get() == nullptr) {
    LOG(ERROR) << "Create Parser error.";
    bo = false;
  }

  LOG(PRINT) << "Initialize Parser successfully.";

  // Read problem to get max_feature and num_field
  scoped_ptr<Reader> reader(CreateReader());
  // Read training file
  reader->Initialize(GetHyperParam()->train_set_file,
                     GetHyperParam()->batch_size,
                     GetParser().get(),
                    GetHyperParam()->model_type);
  //printf("a\n");
  ReadProblem(reader.get(),
               &(GetHyperParam()->max_feature),
               &(GetHyperParam()->num_field));
  // Read test set
  //printf("b\n");
  reader->Initialize(GetHyperParam()->test_set_file,
                     GetHyperParam()->batch_size,
                     GetParser().get(),
                     GetHyperParam()->model_type);
  ReadProblem(reader.get(),
              &(GetHyperParam()->max_feature),
              &(GetHyperParam()->num_field));

  if (GetHyperParam()->model_type == FFM) {
    GetHyperParam()->num_param = (GetHyperParam()->max_feature)
      * (1+GetHyperParam()->num_factor*GetHyperParam()->num_field);
  } else if (GetHyperParam()->model_type == FM) {
    GetHyperParam()->num_param = (GetHyperParam()->max_feature)
      * (1 + GetHyperParam()->num_factor);
    printf("num_param is %u, max_f %u, num_fac %u\n", GetHyperParam()->num_param, 
                                                         GetHyperParam()->max_feature,
                                                         GetHyperParam()->num_factor);
  } else {
    GetHyperParam()->num_param = GetHyperParam()->max_feature;
  }

  LOG(PRINT) << "Read problem successfully.";

  // Create the Model
  if (GetHyperParam()->is_train) {
    GetModel().reset(new Model(GetHyperParam()->num_param,
  	                           GetHyperParam()->updater,
  	                           IfGaussian()));
  } else { // Load model from a checkpoint file
    GetModel().reset(new Model(GetHyperParam()->model_checkpoint_file,
                               GetHyperParam()->updater));
  }
  if (GetModel().get() == nullptr) {
  	LOG(ERROR) << "Create Model error.";
  	bo = false;
  }

  LOG(PRINT) << "Initialize model parameters successfully.";

  // Create the Loss
  GetLoss().reset(CreateLoss());
  if (GetLoss().get() == nullptr) {
  	LOG(ERROR) << "Create Loss error.";
  	bo = false;
  }
  GetLoss()->Initialize(*(GetHyperParam().get()));

  LOG(PRINT) << "Initialize Loss successfully.";

  // Create the Updater
  if (GetHyperParam()->is_train) {
    GetUpdater().reset(CreateUpdater());
    if (GetUpdater().get() == nullptr) {
  	  LOG(ERROR) << "Create Updater error.";
  	  bo = false;
    }
    GetUpdater()->Initialize(*(GetHyperParam().get()));

    LOG(PRINT) << "Initialize Updater successfully.";
  }

  // Create the Validator
  if (GetHyperParam()->is_train) {
    GetValidator().reset(new Validator);
    if (GetValidator().get() == nullptr) {
      LOG(ERROR) << "Create Validator error.";
    }
    GetValidator()->Initialize(GetLoss().get());

    LOG(PRINT) << "Initialize Validator successfully.";
  }

  // Create the FileSpliter
  if (GetHyperParam()->is_train) {
    if (GetHyperParam()->cross_validation) {
      GetFileSpliter().reset(new FileSpliter);
      if (GetFileSpliter().get() == nullptr) {
        LOG(ERROR) << "Create FileSpliter error.";
        bo = false;
      }
      LOG(PRINT) << "Initialize FileSpliter successfully.";
    }
  }

  return bo;
}

void Finalize() {

  LOG(PRINT) << "Finalize successfully.";
}

// Read problem to get the max_feature and num_field
void ReadProblem(Reader* reader, index_t* max_feature, int* num_field) {
  index_t tmp_feature = 0;
  int tmp_field = 0;
  DMatrix* matrix = nullptr;
  //printf("read1\n");
  for (;;) {
    int record_num = reader->Samples(matrix);
    if (record_num == 0) {
      reader->GoToHead();
      break;
    }
     //printf("read1.5\n");
    for (size_t i = 0; i < matrix->row_len; ++i) {
      SparseRow* row = matrix->row[i];
      if (row->id > tmp_feature) {
          tmp_feature = row->id;
      }
    }
  }
   //printf("read2\n");
  // ceil for AVX
  *max_feature = ((tmp_feature + 8) / 8.0) * 8;
  *num_field = tmp_field;
}

//------------------------------------------------------------------------------
// Train model
//------------------------------------------------------------------------------
void StartTrainWork() {
  // Split trainning file into K folds if using cross-validation
  if (GetHyperParam()->cross_validation) {
    GetFileSpliter()->split(GetHyperParam()->train_set_file,
                            GetHyperParam()->num_folds);
    LOG(PRINT) << "Split file successfully.";
  }

  // We train K times if we using a K-folds cross-validation
  int train_num = GetHyperParam()->cross_validation ?
                  GetHyperParam()->num_folds : 1;

  // Init file list
  vector<string> file_list;
  if (GetHyperParam()->cross_validation) {
    file_list.resize(train_num);
    for (int i = 0; i < train_num; ++i) {
      file_list[i] = StringPrintf("%s_%d",
                                  GetHyperParam()->train_set_file.c_str(),
                                  i);
    }
  } else {
    file_list.resize(2);
    file_list[0] = GetHyperParam()->train_set_file;
    file_list[1] = GetHyperParam()->test_set_file;
  }

  // Init Reader
  vector<Reader*> reader_list;
  if (GetHyperParam()->cross_validation) {
    reader_list.resize(train_num);
    for (int i = 0; i < train_num; ++i) {
      reader_list[i] = CreateReader();
      reader_list[i]->Initialize(file_list[i],
                                 GetHyperParam()->batch_size,
                                 GetParser().get(),
                                 GetHyperParam()->model_type);
    }
  } else {
    reader_list.resize(2);
    for (int i = 0; i < 2; ++i) {
      reader_list[i] = CreateReader();
      reader_list[i]->Initialize(file_list[i],
                                 GetHyperParam()->batch_size,
                                 GetParser().get(),
                                 GetHyperParam()->model_type);
    }
  }

  LOG(PRINT) << "Initialize Reader successfully.";

  // Train
  if (GetHyperParam()->cross_validation) {
    CVTrain(reader_list, train_num);
    // Clear the tmp files
    for (int i = 0; i < train_num; ++i) {
      RemoveFile(file_list[i].c_str());
    }
  } else {
    Train(reader_list);
  }
}

//------------------------------------------------------------------------------
// Train without Cross-validation
//------------------------------------------------------------------------------
void Train(const vector<Reader*>& reader_list) {
  LOG(PRINT) << "Start to train model.";
  real_t tmp_loss = kFloatMax;
  std::vector<real_t> tmp_hash_map;
  DMatrix* matrix = nullptr;
  // train loop
  int count = 0;
  for (;;) {
    int record_num = reader_list[0]->Samples(matrix);
    if (record_num == 0) { // end of file
      reader_list[0]->GoToHead();
      // Evaluate current loss
      real_t current_loss =
          GetValidator()->Validate(GetModel().get(),
                                   reader_list[0],
                                   reader_list[1],
                                   count);
      // Using early stopping
      if (GetHyperParam()->early_stop && current_loss > tmp_loss) {
        LOG(PRINT) << "Early stop at iteration " << count
                   << " / " << GetHyperParam()->num_iteration;
        GetModel()->Loadweight(tmp_hash_map);
        break;
      }
      // End of iteration
      if (++count >= GetHyperParam()->num_iteration) {
        break;
      }
      // Save current model
      tmp_loss = current_loss;
      if (GetHyperParam()->early_stop) {
        GetModel()->Saveweight(tmp_hash_map);
      }
      continue;
    }
    // Calc loss and update model parameter
    GetLoss()->CalcGrad(matrix,
                        GetModel().get(),
                        GetUpdater().get());
  }
  // Dump model to disk file
  GetModel()->SaveModel(GetHyperParam()->model_checkpoint_file);
}

//------------------------------------------------------------------------------
// Train with Cross-validation
//------------------------------------------------------------------------------
void CVTrain(const vector<Reader*>& reader_list, int train_num) {
  LOG(PRINT) << "Start to cross validation.";
  DMatrix* matrix = nullptr;
  real_t average_loss = 0.0;
  // K folds
  for (int k = 0; k < train_num; ++k) {
    LOG(PRINT) << "K folds: " << k << "/" << train_num;
    Reader* validate_reader = reader_list[k];
    // Reset current model
    GetModel()->Reset(IfGaussian());
    int reader_id = 0;
    // Train loop
    int count = 0;
    for (;;) {
      // find reader_id
      if (reader_id == k) {
        ++reader_id;
        continue;
      } else if (reader_id >= train_num) {
        reader_id = 0;
        continue;
      }
      int record_num = reader_list[reader_id]->Samples(matrix);
      if (record_num == 0) { // End of file
        reader_list[reader_id]->GoToHead();
        reader_id++;
        if (++count > GetHyperParam()->num_iteration) {
          break;
        }
        continue;
      }
      // Calc loss and update model
      GetLoss()->CalcGrad(matrix,
                          GetModel().get(),
                          GetUpdater().get());
    }
    // loss for the kth test set
    validate_reader->GoToHead();
    real_t current_loss = GetValidator()->Validate(GetModel().get(),
                                                   validate_reader);
    LOG(PRINT) << "Loss value for the "<< k << "th test set : "
              << current_loss;
    average_loss += current_loss;
  }
  average_loss /= GetHyperParam()->num_folds;
  LOG(PRINT) << "The average loss is : " << average_loss;
}

//------------------------------------------------------------------------------
// Predict
//------------------------------------------------------------------------------

void StartPredictWork() {
  scoped_ptr<Reader> reader(CreateReader());
  reader->Initialize(GetHyperParam()->test_set_file,
                     GetHyperParam()->batch_size,
                     GetParser().get(),
                     GetHyperParam()->model_type);
  LOG(PRINT) << "Start predication work.";
  DMatrix* matrix = nullptr;
  std::vector<real_t> pred;
  FILE* file = OpenFileOrDie("./result.txt", "w");
  while (reader->Samples(matrix)) {
    if (pred.size() != matrix->row_len) {
      pred.resize(matrix->row_len);
    }
    GetLoss()->Predict(matrix, GetModel().get(), pred);
    if (GetHyperParam()->sigmoid) {
      SigmoidTrans(pred);
    }
    WritePredToFile(pred, file);
  }
  Close(file);
  LOG(PRINT) << "========= Write result in ./result.txt ===========";
}

// Write the prediction result to disk file
void WritePredToFile(const std::vector<real_t>& pred, FILE* file) {
  std::string str_label;
  for (int i = 0; i < pred.size(); ++i) {
    SStringPrintf(&str_label, "%f\n", pred[i]);
    WriteDataToDisk(file, str_label.c_str(), str_label.size());
  }
}

// Given the original prediction result,
// return the new result transformed by the sigmoid function.
void SigmoidTrans(std::vector<real_t>& pred) {
  CHECK_GT(pred.size(), 0);
  for (size_t i = 0; i < pred.size(); ++i) {
    pred[i] = fastsigmoid(pred[i]); /* from math.h */
  }
}

} // namespace f2m
