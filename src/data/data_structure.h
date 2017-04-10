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

This file defines the basic data structures used by f2m.
*/

#ifndef F2M_DATA_DATA_STRUCTURE_H_
#define F2M_DATA_DATA_STRUCTURE_H_

#include <vector>

#include "src/base/common.h"
#include "src/base/stl-util.h"

namespace f2m {

static size_t max_row_len = 1024 * 1024;

//------------------------------------------------------------------------------
// We use 32 bits float to store real number, such as the model
// parameters and the gradients during computation.
//------------------------------------------------------------------------------
typedef float real_t;

//------------------------------------------------------------------------------
// We use 32 bits unsigned int to store the index of model parameters.
//------------------------------------------------------------------------------
typedef uint32 index_t;

//------------------------------------------------------------------------------
// Indicate which Ml algorithm we use in current task.
//------------------------------------------------------------------------------
enum ModelType {
  LR,
  FM,
  FFM,
  LINEAR,
  SVM
};

//------------------------------------------------------------------------------
// Indicate which ML task we are solving, including: Binary classification,
// Multiple class classification, as well as Regression.
// Classification model: LR, FM, FFM, SVM
// Regression model: LINEAR, FM, FFM
//------------------------------------------------------------------------------
enum TaskType {
  Binary,
  MultiClass,
  Regression
};

//------------------------------------------------------------------------------
// Indicate which Regularizer we use in current task.
//------------------------------------------------------------------------------
enum RegularType {
  L1,
  L2,
  NONE
};

//------------------------------------------------------------------------------
// Indicate which Updater we use in current task.
//------------------------------------------------------------------------------
enum UpdaterType {
  SGD,
  AdaGrad,
  AdaDelta,
  Momentum,
  RMSprop,
  Adam
};

//------------------------------------------------------------------------------
// Indicate which Parser we use in current task.
//------------------------------------------------------------------------------
enum ParserType {
  LibSVM,
  LibFFM,
  CSV
};

//------------------------------------------------------------------------------
// SparseRow is used to store one line data of the DMatrix.
// Note that we do not use map<int, float> to store sparse entry because of
// its poor performance. Instead, we use double-vector, in which the feature
// and its index are stored in two vectors. For the FFM task, we also need
// anther vector to store the field.
//------------------------------------------------------------------------------
struct SparseRow {
  // On default the 'field' vector is empty.
  explicit SparseRow(size_t length, bool has_field = false)
    : X(length, 0.0), idx(length, 0),id(0), if_has_field(has_field) {
    // for ffm task
    if (if_has_field) {
      field.resize(length, 0);
    }
    column_len = length;
  }

  // Resize current row. Note that we only invoke the resize()
  // when the new length is larger than current length.
  void Resize(size_t new_length) {
    CHECK_GE(new_length, 0);
    if (column_len < new_length) {
      X.resize(new_length, 0.0);
      idx.resize(new_length, 0);
      if (if_has_field) {
        field.resize(new_length, 0);
      }
    }
    column_len = new_length;
  }

  // Copy data from one SparseRow to another.
  void CopyFrom(SparseRow* row) {
    CHECK_NOTNULL(row);
    if_has_field = row->if_has_field;
    id = row->id;
    Resize(row->column_len);
    std::copy(row->X.begin(), row->X.end(), this->X.begin());
    std::copy(row->idx.begin(), row->idx.end(), this->idx.begin());
    if (if_has_field) {
      std::copy(row->field.begin(), row->field.end(), this->field.begin());
    }
  }

  std::vector<real_t> X;       // Storing the feature value.
  std::vector<index_t> idx;    // Stroing the index of each feature.
  std::vector<index_t> field;  // Storing the field value.
  size_t column_len;           // Storing the size of current row.
  int id;
  bool if_has_field;           // for ffm ?
};

//------------------------------------------------------------------------------
// DMatrix (data matrix) is used to store a batch of trainning dataset.
// For many large-scale Ml problems, we can not load all the trainning data
// into memory at once. So we can load a small batch of dataset into the
// DMatrix in each iteration.
//------------------------------------------------------------------------------
struct DMatrix {
  // Constructor and Destructor
  DMatrix() {  
    Y.clear();
  }
  ~DMatrix() { Release(); }

  explicit DMatrix(size_t length)
    : row(length, nullptr),
      row_len(length),
      can_release(false) { 
    Y.clear();
  }

  // Reset current row length
  void Setlength(size_t new_length) {
    CHECK_GE(new_length, 0);
    row_len = new_length;
  }

  // Resize current DMatrix
  void Resize(size_t num_lines) {
    CHECK_GE(num_lines, 0);
    Release();
    row_len = num_lines;
    row.resize(num_lines, nullptr);
    can_release = false;
  }

  // Initialize the row pointers. On default the field is empty.
  void InitSparseRow(bool has_field = false) {
    // Pointers have been initialized
    if (can_release) {
      LOG(FATAL) << "Attempt to initialize the pointers twice.";
    }
    this->has_field = has_field;
    for (size_t i = 0; i < row_len; ++i) {
      if (row[i] == nullptr) {
        row[i] = new SparseRow(0, has_field);
      }
    }
    can_release = true;
  }

  // Copy data from one buffer to another. Note that we assume that
  // the row_len of original matrix >= that of the new matrix.
  // Suppose that when invoking this function, 
  // the matrix has only one line of Y values.
  void CopyFrom(DMatrix& matrix) {
    CHECK_GE(row_len, matrix.row_len);
    Resize(matrix.row_len);
    InitSparseRow(matrix.has_field);
     std::copy(matrix.Y[0]->begin(), matrix.Y[0]->end(), Y[0]->begin());
    for (size_t i = 0; i < row_len; ++i) {
      row[i]->CopyFrom(matrix.row[i]);
    }
  }

  // Release memory of all SparseRows.
  void Release() {
    // To avoid double free
    if (can_release) {
      STLDeleteElementsAndClear(&row);
      STLDeleteElementsAndClear(&Y);
    }
  }

  // Storing SparseRows. Note that we use pointers here in order to
  // implement zero copy when copying data betweent different DMatrix(s).
  std::vector<SparseRow*> row;
  // Y can be either -1 or 0 (for negetive examples), and
  // can be 1 (for positive examples).
  std::vector<std::vector<real_t>* > Y;
  // for ffm ?
  bool has_field;
  // Row length of current DMatrix.
  size_t row_len;
  // To avoid double free.
  bool can_release;
};

} // namespace f2m

#endif // F2M_DATA_DATA_STRUCTURE_H_
