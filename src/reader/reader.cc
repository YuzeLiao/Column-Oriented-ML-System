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

This file is the implementation of Reader.
*/

#include "src/reader/reader.h"

#include <vector>
#include <string>
#include <algorithm> // for random_shuffle

#include <string.h>

#include "src/base/common.h"
#include "src/base/file_util.h"
#include "src/base/scoped_ptr.h"

static const uint32 kMaxLineSize = 100 * 1024; // 100 KB for one line of data

namespace f2m {

//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_IMPLEMENT_REGISTRY(f2m_reader_registry, Reader);
REGISTER_READER("memory", InmemReader);
//REGISTER_READER("disk", OndiskReader);

//------------------------------------------------------------------------------
// Implementation of InmemReader
//------------------------------------------------------------------------------

InmemReader::~InmemReader() {
  if (file_ptr_ != nullptr) {
    Close(file_ptr_);
  }
  data_buf_.Release();
}

// Pre-load all the data into memory buffer.
void InmemReader::Initialize(const std::string& filename,
                             int num_samples,
                             Parser* parser,
                             ModelType type) {
  CHECK_NE(filename.empty(), true)
  CHECK_GT(num_samples, 0);
  CHECK_NOTNULL(parser);
  filename_ = filename;
  num_samples_ = num_samples;
  parser_ = parser;
  pos_ = 0;
  file_ptr_ = OpenFileOrDie(filename_.c_str(), "r");
  uint64 file_size = GetFileSize(file_ptr_);
  scoped_array<char> buffer;
  try {
    buffer.reset(new char[file_size]);
  } catch (std::bad_alloc&) {
    LOG(FATAL) << "Cannot allocate enough memory for Reader.";
  }
  // Read all the data from file
  uint64 read_size = fread(buffer.get(),
                           1,
                           file_size,
                           file_ptr_);
  CHECK_EQ(read_size, file_size);
  // Initialize the DMatrix buffer
  int num_lines = GetLineNumber(buffer.get(), read_size);
  bool if_has_field = type == FFM ? true : false;
  // Parse each line of data from buffer
  StringList list(num_lines);
  scoped_array<char> line(new char[kMaxLineSize]);
  uint64 start_pos = 0;
  for (int i = 0; i < num_lines; ++i) {
    uint32 line_size = ReadLineFromMemory(line.get(),
                                          buffer.get(),
                                          start_pos,
                                          file_size);
    CHECK_NE(line_size, 0);
    start_pos += line_size;
    line[line_size - 1] = '\0';
    if (line_size > 1 && line[line_size - 2] == '\r') {
      // Handle some txt format in windows or DOS.
      line[line_size - 2] = '\0';
    }
    list[i].assign(line.get());
  }
  // Parse StringList to DMatrix.
  --num_lines;
  int max_lines = atoi(list[num_lines].c_str());
  data_samples_.Resize(max_lines);
  data_samples_.Y.resize(1, NULL);
  data_buf_.Resize(num_lines);
  data_buf_.InitSparseRow(if_has_field);
  sampled_length.clear();
  parser_->Parse(list, data_buf_, sampled_length);
}

uint32 InmemReader::ReadLineFromMemory(char* line,
                                       char* buf,
                                       uint64 start_pos,
                                       uint64 total_len) {
  // End of file
  if (start_pos >= total_len) {
    return 0;
  }
  uint64 end_pos = start_pos;
  while (*(buf + end_pos) != '\n') { ++end_pos; }
  uint32 read_size = end_pos - start_pos + 1;
  if (read_size > kMaxLineSize) {
    LOG(FATAL) << "Encountered a too-long line. Please check the data.";
  }
  memcpy(line, buf + start_pos, read_size);
  return read_size;
}

int InmemReader::GetLineNumber(const char* buf, uint64 buf_size) {
  int num = 0;
  for (uint64 i = 0; i < buf_size; ++i) {
    if (buf[i] == '\n') num++;
  }
  return num;
}

// Smaple data from memory buffer.
int InmemReader::Samples(DMatrix* &matrix) {
  static int now_block = 0;
  int num_lines = sampled_length[now_block];
  if (pos_ == data_buf_.row_len) {
    now_block = 0;
    matrix = NULL;
    return 0;
  }
  data_samples_.Y[0] = data_buf_.Y[now_block];
  for (int i = 0; i < num_lines; ++i) {
    // Copy data between different DMatrix.
    data_samples_.row[i] = data_buf_.row[pos_];
    pos_++;
  }
  now_block++;
  data_samples_.Setlength(num_lines);
  matrix = &data_samples_;
  return num_lines;
}

// Return to the begining of the data buffer.
void InmemReader::GoToHead() { pos_ = 0; }

// Using Min-max normalization
void InmemReader::Normalize(real_t max, real_t min) {
  for (size_t i = 0; i < data_buf_.row_len; ++i) {
    SparseRow* row = data_buf_.row[i];
    for (size_t j = 0; j < row->column_len; ++j) {
      row->X[j] = (row->X[j] - min) / (max - min);
    }
  }
}

//------------------------------------------------------------------------------
// Implementation of OndiskReader.
//------------------------------------------------------------------------------

/*OndiskReader::~OndiskReader() {
  if (file_ptr_ != nullptr) {
    Close(file_ptr_);
  }
  data_samples_.Release();
}

void OndiskReader::Initialize(const std::string& filename,
                              int num_samples,
                              Parser* parser,
                              ModelType type) {
  CHECK_NE(filename.empty(), true);
  CHECK_GT(num_samples, 0);
  CHECK_NOTNULL(parser);
  filename_ = filename;
  num_samples_ = num_samples;
  parser_ = parser;
  file_ptr_ = OpenFileOrDie(filename_.c_str(), "r");
  bool if_has_field = type == FFM ? true : false;
  data_samples_.Resize(num_samples);
  data_samples_.InitSparseRow(if_has_field);
}

// Sample data from disk file.
int OndiskReader::Samples(DMatrix* &matrix) {
  static scoped_array<char> line(new char[kMaxLineSize]);
  static StringList list(num_samples_);
  int num_lines = 0;
  for (int i = 0; i < num_samples_; ++i) {
    if (fgets(line.get(), kMaxLineSize, file_ptr_) == nullptr) {
      // Either ferror or feof.
      if (i == 0) {
        matrix = nullptr;
        return 0;
      }
      break;
    }
    int read_len = strlen(line.get());
    if (line[read_len - 1] != '\n') {
      LOG(FATAL) << "Encountered a too-long line. Please check the data.";
    } else {
      line[read_len - 1] = '\0';
      // Handle the txt format in DOS and windows.
      if (read_len > 1 && line[read_len - 2] == '\r') {
        line[read_len - 2] = '\0';
      }
    }
    list[i].assign(line.get());
    num_lines++;
  }
  // The last data block
  if (num_lines != num_samples_) {
    data_samples_.Setlength(num_lines);
  }
  parser_->Parse(list, data_samples_);
  matrix = &data_samples_;
  return num_lines;
}

// Return to the begining of the file.
void OndiskReader::GoToHead() { fseek(file_ptr_, 0, SEEK_SET); }
*/
} // namespace f2m
