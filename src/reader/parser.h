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

This file defines the Parser class, which parses the Reader's output
to the DMatrix.
*/

#ifndef F2M_READER_PARSER_H_
#define F2M_READER_PARSER_H_

#include <vector>
#include <string>

#include "src/base/common.h"
#include "src/base/class_register.h"
#include "src/data/data_structure.h"

namespace f2m {

typedef std::vector<std::string> StringList;

//------------------------------------------------------------------------------
// Given a StringList, parse it to the DMatrix format.
// Parser is an abstract class, which can be implemented by real Parser
// such as the LibsvmParser and the FFMParser.
// Note that the Parser will add a bias (i.e., 1.0) term in the front of the
// input data (in each line) by default.
//------------------------------------------------------------------------------
class Parser {
 public:
  // Using the " " as splitor by default.
  Parser() { }

  virtual ~Parser() {  }

  void SetSplitor(std::string splitor) { m_splitor = splitor; }

  virtual void Parse(const StringList& list, DMatrix& matrix, std::vector<index_t>& sampled_length) = 0;

 protected:
  std::string m_splitor = " ";  // Identify the spliting character
  StringList m_items;           // To store items divided by the splitor
  StringList m_single_item;     // To store every single item divided by ':'

 private:

  DISALLOW_COPY_AND_ASSIGN(Parser);
};

//------------------------------------------------------------------------------
// LibsvmParser parses the following data format:
// [y1 idx:value idx:value ...]
// [y2 idx:value idx:value ...]
//------------------------------------------------------------------------------
class LibsvmParser : public Parser {
 public:
  LibsvmParser() {  }
  ~LibsvmParser() {  }

  virtual void Parse(const StringList& list, DMatrix& matrix, std::vector<index_t>& sampled_length);

 private:

  DISALLOW_COPY_AND_ASSIGN(LibsvmParser);
};

//------------------------------------------------------------------------------
// FFMParser parses the following data format:
// [y1 field:idx:value field:idx:value ...]
// [y2 field:idx:value field:idx:value ...]
//------------------------------------------------------------------------------
/*class FFMParser : public Parser {
 public:
  FFMParser() {  }
  ~FFMParser() {  }

  virtual void Parse(const StringList& list, DMatrix& matrix);

 private:

  DISALLOW_COPY_AND_ASSIGN(FFMParser);
};

//------------------------------------------------------------------------------
// CSVParser parses the following data format:
// [y1 value value value ...]
// [y2 value value value ...]
//------------------------------------------------------------------------------
class CSVParser : public Parser {
 public:
  CSVParser() { }
  ~CSVParser() { }

  virtual void Parse(const StringList& list, DMatrix& matrix);

 private:

  DISALLOW_COPY_AND_ASSIGN(CSVParser);
};
*/
//------------------------------------------------------------------------------
// Class register
//------------------------------------------------------------------------------
CLASS_REGISTER_DEFINE_REGISTRY(f2m_parser_registry, Parser);

#define REGISTER_PARSER(format_name, parser_name)          \
  CLASS_REGISTER_OBJECT_CREATOR(                           \
      f2m_parser_registry,                                 \
      Parser,                                              \
      format_name,                                         \
      parser_name)

#define CREATE_PARSER(format_name)                         \
  CLASS_REGISTER_CREATE_OBJECT(                            \
      f2m_parser_registry,                                 \
      format_name)

} // namespace f2m

#endif // F2M_READER_PARSER_H_
