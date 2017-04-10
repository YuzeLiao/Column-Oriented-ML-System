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

This file defines flags used by f2m to accept instructions.
*/

#include "src/train/flags.h"

#include <sys/utsname.h>          // For uname
#include <time.h>
#include <unistd.h>

#include <set>

#include "src/base/common.h"
#include "src/base/stringprintf.h"
#include "src/data/data_structure.h"

namespace f2m {

DEFINE_bool(f2m_is_train, true, "For trainning or predicting. "
                                "This flag is set to true by default.");

DEFINE_string(f2m_task_type, "binary", "Indicate what machine learning task "
                                       "we are solving, including: binary, "
                                       "multi-class, and regression.");

DEFINE_string(f2m_model_type, "", "Indicate which machine learning model "
                                  "we use in current task, including: "
                                  "'lr', 'fm', 'ffm', 'linear', 'svm'. ");

DEFINE_bool(f2m_is_sparse, false, "Storing gradient a dense or sparse way. "
                                  "By default this flag is set to false, "
                                  "i.e., in a dense model.");

DEFINE_double(f2m_learning_rate, 0.01, "Control the learning step. "
                                       "By default this flag is set to 0.01.");

DEFINE_string(f2m_file_format, "libsvm", "Indicate the file format, including: "
                                         "'libsvm', 'libffm', 'csv'. ");

DEFINE_int32(f2m_num_factor, 10, "The number of latent factors. This flag is "
                                 "optional and is only used by the fm task "
                                 "and the ffm task. Default = 10.");

DEFINE_string(f2m_updater, "sgd", "Indicate which updater we use in current "
                                  "task, including: 'sgd', 'adagard', "
                                  "'adadelta', 'momentum', 'rmsprop', and "
                                  "'adam'. We use the sgd updater by defualt.");

DEFINE_double(f2m_decay_rate, 0.9, "The decay factor used by updater. "
                                   "This flag is set to 0.9 by default.");

DEFINE_double(f2m_second_decay_rate, 0.9, "The second decay factor used by "
                                          "updater. This flag is set to 0.9 "
                                          "by defaul.");

DEFINE_double(f2m_regu_lambda, 0.01, "Lambda term for regularizer. "
                                     "We set this flag to 0.01 by default.");

DEFINE_string(f2m_regu_type, "l2", "Indicate which regularizer we use in current "
                                   "task, including: 'l1', 'l2', and 'none'. "
                                   "By default we use the L2 regularizer.");

DEFINE_string(f2m_train_set_file, "", "Filename of the trainning dataset.");

DEFINE_string(f2m_test_set_file, "", "Filename of the testing dataset.");

DEFINE_string(f2m_model_checkpoint_file, "/tmp/model_ck", "File for saving model checkpoint.");

DEFINE_int32(f2m_num_iteration, 50, "Number of iterations. Default = 50");

DEFINE_bool(f2m_cross_validation, false, "If using cross validation. This flag "
                                         "is set to false by default.");

DEFINE_int32(f2m_num_folds, 10, "Number of folds if using cross-validation. "
                                "This flag is set to 10 by default.");

DEFINE_bool(f2m_in_memory_trainning, true, "Trainning model in memory or on "
                                           "disk. By default we set this flag "
                                           "to true for in-memory trainning.");

DEFINE_int32(f2m_batch_size, 1000, "Mini-batch size in each iteration. "
                                   "We set this flag to 1000 by defaul.");

DEFINE_bool(f2m_early_stop, false, "If trainning model using early-stop. "
                                  "By default we set this flag to false.");

DEFINE_bool(f2m_sigmoid, false, "If transfer result using sigmoid function.");

DEFINE_string(f2m_log_filebase, "./log/log", "The real log filename is log_filebase "
                                    "appended date, time, proesses_id, log "
                                    "type and etc.");

//------------------------------------------------------------------------------
// Check the correctness of flags.
//------------------------------------------------------------------------------
bool ValidateCommandLineFlags() {
  // The return value.
  bool flags_valid = true;

  // Check the model type.
  std::string model[] = {"lr", "fm", "ffm", "linear", "svm"};
  std::set<std::string> model_list(model, model+5);
  if (model_list.find(FLAGS_f2m_model_type) == model_list.end()) {
    LOG(ERROR) << "Model type can only be 'lr', 'fm', 'ffm', "
               << "'linear', or 'svm'.";
    flags_valid = false;
  }

  // Check the task type.
  std::string task_type[] = {"binary", "multi-class", "regression"};
  std::set<std::string> task_list(task_type, task_type+3);
  if (task_list.find(FLAGS_f2m_task_type) == task_list.end()) {
    LOG(ERROR) << "Task type can only be 'binary', 'multi-class', "
               << "or 'regression'.";
    flags_valid = false;
  }

  // The learning rate must be greater than 0.
  if (FLAGS_f2m_learning_rate <= 0.0) {
    LOG(ERROR) << "The learning rate must be greater than 0.0";
    flags_valid = false;
  }

  // Check the file format
  std::string file_format[] = {"libsvm", "libffm", "csv"};
  std::set<std::string> format_list(file_format, file_format+3);
  if (format_list.find(FLAGS_f2m_file_format) == format_list.end()) {
    LOG(ERROR) << "File format can only be 'libsvm', 'libffm', or 'csv'.";
    flags_valid = false;
  }

  // The num_factor must be greater than 0.
  if (FLAGS_f2m_model_type == "ffm" || FLAGS_f2m_model_type == "fm") {
    if (FLAGS_f2m_num_factor <= 0) {
      LOG(ERROR) << "The num_factor must be greater than 0";
      flags_valid = false;
    }
  }

  // Check the updater type.
  std::string updater[] = {"sgd", "adagrad", "adadelta", "momentum",
                           "rmsprop", "adam"};
  std::set<std::string> updater_list(updater, updater+6);
  if (updater_list.find(FLAGS_f2m_updater) == updater_list.end()) {
    LOG(ERROR) << "Updater type can only be 'sgd', 'adagrad', 'adadelta'"
               << "'momentum', 'rmsprop', or 'adam'.";
    flags_valid = false;
  }

  // The decay_rate must be greater than 0.0.
  if (FLAGS_f2m_updater == "rmsprop" || FLAGS_f2m_updater == "momentum" ||
          FLAGS_f2m_updater == "adam" || FLAGS_f2m_updater == "adadelta") {
    if (FLAGS_f2m_decay_rate <= 0.0) {
      LOG(ERROR) << "The decay_rate must be greater than 0.0.";
      flags_valid = false;
    }
  }

  // The second_decay_rate must be greater than 0.0
  if (FLAGS_f2m_updater == "adam" && FLAGS_f2m_decay_rate <= 0.0) {
    LOG(ERROR) << "The second_decay_rate must be greater than 0.0.";
    flags_valid = false;
  }

  // The regu_lambda must be greater than or equal to 0.0.
  if (FLAGS_f2m_regu_lambda < 0.0) {
    LOG(ERROR) << "The regu_lambda must be greater than or equal to 0.0.";
    flags_valid = false;
  }

  // Check the regu_type.
  std::string regu[] = {"l1", "l2", "none"};
  std::set<std::string> regu_list(regu, regu+3);
  if (regu_list.find(FLAGS_f2m_regu_type) == regu_list.end()) {
    LOG(ERROR) << "The regu_type can only be 'l1', 'l2', or 'none'.";
    flags_valid = false;
  }

  // The trainning data file cannot be empty.
  if (FLAGS_f2m_train_set_file.empty()) {
    LOG(ERROR) << "The trainning data file cannot be empty.";
    flags_valid = false;
  }

  // The testing data file cannot be empty.
  if (FLAGS_f2m_test_set_file.empty()) {
    LOG(ERROR) << "The testing data file cannot be empty.";
    flags_valid = false;
  }

  // The model checkpoint file cannot be empty.
  if (FLAGS_f2m_model_checkpoint_file.empty()) {
    LOG(ERROR) << "The model checkpoint file cannot be empty.";
    flags_valid = false;
  }

  // The num_iteration must be greater than 0.
  if (FLAGS_f2m_num_iteration <= 0) {
    LOG(ERROR) << "The num_iteration must be greater than 0.";
    flags_valid = false;
  }

  // The num_folds must be greater than 0.
  if (FLAGS_f2m_cross_validation && FLAGS_f2m_num_folds <= 0) {
    LOG(ERROR) << "The num_folds must be greater than 0.";
    flags_valid = false;
  }

  // The batch size must be greater than 0.
  if (FLAGS_f2m_batch_size <= 0) {
    LOG(ERROR) << "The batch_size must be greater than 0.";
    flags_valid = false;
  }

  // The log_filebase cannot be empty.
  if (FLAGS_f2m_log_filebase.empty() == true) {
    LOG(ERROR) << "The log_filebase cannot be empty.";
    flags_valid = false;
  }

  return flags_valid;
}

//------------------------------------------------------------------------------
// Set the hyper_param
//------------------------------------------------------------------------------
void SetHyperParam(HyperParam& hyper_param) {
  // train or predict
  hyper_param.is_train = FLAGS_f2m_is_train;
  if (FLAGS_f2m_task_type == "binary") {
    hyper_param.task_type = Binary;
  }
  else if (FLAGS_f2m_task_type == "multi-class") {
    hyper_param.task_type = MultiClass;
  }
  else if (FLAGS_f2m_task_type == "regression") {
    hyper_param.task_type = Regression;
  }
  else LOG(FATAL) << "Task type error: " << FLAGS_f2m_task_type;
  // model type
  if (FLAGS_f2m_model_type == "lr") hyper_param.model_type = LR;
  else if (FLAGS_f2m_model_type == "fm") hyper_param.model_type = FM;
  else if (FLAGS_f2m_model_type == "ffm") hyper_param.model_type = FFM;
  else if (FLAGS_f2m_model_type == "linear") hyper_param.model_type = LINEAR;
  else if (FLAGS_f2m_model_type == "svm") hyper_param.model_type = SVM;
  else LOG(FATAL) << "Model type error: " << FLAGS_f2m_model_type;
  // sparse
  hyper_param.is_sparse = FLAGS_f2m_is_sparse;
  // learning_rate
  hyper_param.learning_rate = FLAGS_f2m_learning_rate;
  // parser
  if (FLAGS_f2m_file_format == "libsvm") hyper_param.parser = LibSVM;
  else if (FLAGS_f2m_file_format == "libffm") hyper_param.parser = LibFFM;
  else if (FLAGS_f2m_file_format == "csv") hyper_param.parser = CSV;
  else LOG(FATAL) << "File format error: " << FLAGS_f2m_file_format;
  // num factor
  hyper_param.num_factor = FLAGS_f2m_num_factor;
  // updater type
  if (FLAGS_f2m_updater == "sgd") hyper_param.updater = SGD;
  else if (FLAGS_f2m_updater == "adagrad") hyper_param.updater = AdaGrad;
  else if (FLAGS_f2m_updater == "adadelta") hyper_param.updater = AdaDelta;
  else if (FLAGS_f2m_updater == "momentum") hyper_param.updater = Momentum;
  else if (FLAGS_f2m_updater == "rmsprop") hyper_param.updater = RMSprop;
  else if (FLAGS_f2m_updater == "adam") hyper_param.updater = Adam;
  else LOG(FATAL) << "Updater type error: " << FLAGS_f2m_updater;
  // decay rate
  hyper_param.decay_rate = FLAGS_f2m_decay_rate;
  // second decay rate
  hyper_param.second_decay_rate = FLAGS_f2m_second_decay_rate;
  // regu lambda
  hyper_param.regu_lambda = FLAGS_f2m_regu_lambda;
  // regu type
  if (FLAGS_f2m_regu_type == "l1") hyper_param.regu_type = L1;
  else if (FLAGS_f2m_regu_type == "l2") hyper_param.regu_type = L2;
  else if (FLAGS_f2m_regu_type == "none") hyper_param.regu_type = NONE;
  else LOG(FATAL) << "Regularizer error: " << FLAGS_f2m_regu_type;
  // trainning set filename
  hyper_param.train_set_file = FLAGS_f2m_train_set_file;
  // test set filename
  hyper_param.test_set_file = FLAGS_f2m_test_set_file;
  // model checkpoint file
  hyper_param.model_checkpoint_file = FLAGS_f2m_model_checkpoint_file;
  // number iteration
  hyper_param.num_iteration = FLAGS_f2m_num_iteration;
  // cross validation
  hyper_param.cross_validation = FLAGS_f2m_cross_validation;
  // number folds
  hyper_param.num_folds = FLAGS_f2m_num_folds;
  // in-memory or on disk trainning
  hyper_param.in_memory_trainning = FLAGS_f2m_in_memory_trainning;
  // mini-batch size
  hyper_param.batch_size = FLAGS_f2m_batch_size;
  // early stop
  hyper_param.early_stop = FLAGS_f2m_early_stop;
  // sigmoid
  hyper_param.sigmoid = FLAGS_f2m_sigmoid;
}

//------------------------------------------------------------------------------
// For the log in our system
//------------------------------------------------------------------------------

std::string GetUserName() {
  const char* username = getenv("USER");
  return username != NULL ? username : getenv("USERNAME");
}

std::string PrintCurrentTime() {
  time_t current_time = time(NULL);
  struct tm broken_down_time;
  CHECK(localtime_r(&current_time, &broken_down_time) == &broken_down_time);
  return StringPrintf("%04d%02d%02d-%02d%02d-%02d",
                      1900 + broken_down_time.tm_year,
                      1 + broken_down_time.tm_mon,
                      broken_down_time.tm_mday, broken_down_time.tm_hour,
                      broken_down_time.tm_min,  broken_down_time.tm_sec);
}

std::string LogFilebase() {
  // log_filebase := FLAGS_f2m_log_filebase +
  //                 username +
  //                 date_time +
  //                 process_id
  std::string filename_prefix;
  SStringPrintf(&filename_prefix,
                "%s-%s-%s-%u",
                FLAGS_f2m_log_filebase.c_str(),
                GetUserName().c_str(),
                PrintCurrentTime().c_str(),
                getpid());

  return filename_prefix;
}

//------------------------------------------------------------------------------
// Create object
//------------------------------------------------------------------------------
Reader* CreateReader() {
  Reader* reader = nullptr;
  std::string reader_type =
    FLAGS_f2m_in_memory_trainning ? "memory" : "disk";
  reader = CREATE_READER(reader_type.c_str());
  if (reader == nullptr) {
    LOG(ERROR) << "Cannot create Reader: " << reader_type;
  }
  return reader;
}

Parser* CreateParser() {
  Parser* parser = nullptr;
  std::string parser_type = FLAGS_f2m_file_format;
  parser = CREATE_PARSER(parser_type.c_str());
  if (parser == nullptr) {
    LOG(ERROR) << "Cannot create Parser: " << parser_type;
  }
  return parser;
}

Loss* CreateLoss() {
  Loss* loss = nullptr;
  std::string loss_type = FLAGS_f2m_model_type;
  loss = CREATE_LOSS(loss_type.c_str());
  if (loss == nullptr) {
    LOG(ERROR) << "Cannot create Loss: " << loss_type;
  }
  return loss;
}

Updater* CreateUpdater() {
  Updater* updater = nullptr;
  std::string updater_type = FLAGS_f2m_updater;
  updater = CREATE_UPDATER(updater_type.c_str());
  if (updater == nullptr) {
    LOG(ERROR) << "Cannot create updater: " << updater_type;
  }
  return updater;
}

bool IfGaussian() {
  return (FLAGS_f2m_model_type == "fm" ||
          FLAGS_f2m_model_type == "ffm") ? true : false;
}

} // namespace f2m
