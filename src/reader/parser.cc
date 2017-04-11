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

This file is the implementation of Parser.
*/

#include "src/reader/parser.h"

#include <stdlib.h>

#include "src/base/common.h"
#include "src/base/split_string.h"
#include "src/data/data_structure.h"

namespace f2m {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(f2m_parser_registry, Parser);
REGISTER_PARSER("libsvm", LibsvmParser);
//REGISTER_PARSER("libffm", FFMParser);
//REGISTER_PARSER("csv", CSVParser);

//------------------------------------------------------------------------------
// LibsvmParser parses the following data format:
// [y1 idx:value idx:value ...]
// [y2 idx:value idx:value ...]
//------------------------------------------------------------------------------

// This can only be used for in-memory trainning now.
void LibsvmParser::Parse(const StringList& list, DMatrix& matrix, std::vector<index_t>& sampled_length) {
  CHECK_GE(list.size(), 0);
  CHECK_GE(matrix.row_len, 0);
  size_t row_len = matrix.row_len;
  index_t row_pos = 0;
  for (size_t i = 0; i < row_len; ++i, ++row_pos) {
    m_items.clear();
    SplitStringUsing(list[i], m_splitor.c_str(), &m_items);
    int len = m_items.size();
    if (len == 1) {
      int row_length = atoi(m_items[0].c_str());
      sampled_length.push_back(row_length);
      matrix.row_len--;
      ++i;
      m_items.clear();
      SplitStringUsing(list[i], m_splitor.c_str(), &m_items);
      int len_ = m_items.size();
      matrix.row[row_pos]->Resize(len_);
      matrix.row[row_pos]->id = 0;
      std::vector<real_t>* tmp_y = new std::vector<real_t>(len_, 0.0);
      for (int j = 0; j < len_; ++j) {
        (*tmp_y)[j] = atof(m_items[j].c_str());
        matrix.row[row_pos]->idx[j] = j;
        matrix.row[row_pos]->X[j] = 1.0; 
      }
      matrix.Y.push_back(tmp_y);
      continue;
    }
    CHECK_NOTNULL(matrix.row[row_pos]);
    matrix.row[row_pos]->Resize(--len);
    // add bias term.
    matrix.row[row_pos]->id = atoi(m_items[0].c_str());
    for (int j = 0; j < len; ++j) {
      m_single_item.clear();
      SplitStringUsing(m_items[j + 1], ":", &m_single_item);
      CHECK_EQ(m_single_item.size(), 2);
      matrix.row[row_pos]->idx[j] = atoi(m_single_item[0].c_str());
      matrix.row[row_pos]->X[j] = atof(m_single_item[1].c_str());
    }
  }
}
/*
//------------------------------------------------------------------------------
// FFMParser parses the following data format:
// [y1 field:idx:value field:idx:value ...]
// [y2 field:idx:value field:idx:value ...]
//------------------------------------------------------------------------------
void FFMParser::Parse(const StringList& list, DMatrix& matrix) {
  CHECK_GE(list.size(), 0);
  CHECK_GE(matrix.row_len, 0);
  size_t row_len = matrix.row_len;
  for (size_t i = 0; i < row_len; ++i) {
    m_items.clear();
    SplitStringUsing(list[i], m_splitor.c_str(), &m_items);
    int len = m_items.size();
    matrix.Y[i] = atof(m_items[0].c_str());
    CHECK_NOTNULL(matrix.row[i]);
    // add bias term
    matrix.row[i]->Resize(len);
    matrix.row[i]->field[0] = 0;
    matrix.row[i]->idx[0] = 0;
    matrix.row[i]->X[0] = 1.0;
    for (int j = 1; j < len; ++j) {
      m_single_item.clear();
      SplitStringUsing(m_items[j], ":", &m_single_item);
      CHECK_EQ(m_single_item.size(), 3);
      matrix.row[i]->field[j] = atoi(m_single_item[0].c_str());
      matrix.row[i]->idx[j] = atoi(m_single_item[1].c_str());
      matrix.row[i]->X[j] = atof(m_single_item[2].c_str());
    }
  }
}

//------------------------------------------------------------------------------
// CSVParser parses the following data format:
// [y1 value value value ...]
// [y2 value value value ...]
//------------------------------------------------------------------------------
void CSVParser::Parse(const StringList& list, DMatrix& matrix) {
  CHECK_GE(list.size(), 0);
  CHECK_GE(matrix.row_len, 0);
  size_t row_len = matrix.row_len;
  for (size_t i = 0; i < row_len; ++i) {
    m_items.clear();
    SplitStringUsing(list[i], m_splitor.c_str(), &m_items);
    matrix.Y[i] = atof(m_items[0].c_str());
    // Get real length of current row
    int len = 0;
    for (int j = 1; j < m_items.size(); ++j) {
      if (atof(m_items[j].c_str()) != 0) {
        ++len;
      }
    }
    CHECK_NOTNULL(matrix.row[i]);
    matrix.row[i]->Resize(len+1); // add a bias
    matrix.row[i]->idx[0] = 0;
    matrix.row[i]->X[0] = 1.0;
    int k = 1;
    for (int j = 1; j < m_items.size(); ++j) {
      real_t value = atof(m_items[j].c_str());
      if (value != 0) {
        matrix.row[i]->idx[k] = j;
        matrix.row[i]->X[k] = value;
        ++k;
      }
    }
  }
}*/

} // namespace f2m
