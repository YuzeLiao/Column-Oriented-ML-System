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

This file is the program entry.
*/

#include "src/train/flags.h"
#include "src/train/train.h"
#include "src/train/train.cc"
#include "gflags/gflags.h"

/*------------------------------------------------------------------------------
       _
      | |
 __  _| |     ___  __ _ _ __ _ __
 \ \/ / |    / _ \/ _` | '__| '_ \
  >  <| |___|  __/ (_| | |  | | | |
 /_/\_\______\___|\__,_|_|  |_| |_|

*------------------------------------------------------------------------------*/
void PrintLogo() {
  std::cout << "---------------------------------------------\n"
            << "      _\n"
            << "     | |\n"
            << "__  _| |     ___  __ _ _ __ _ __\n"
            << "\\ \\/ / |    / _ \\/ _` | '__| '_ \\ \n"
            << " >  <| |___|  __/ (_| | |  | | | |\n"
            << "/_/\\_\\_____/\\___|\\__,_|_|  |_| |_|\n\n"
            << "xLearn   -- 0.10 Version --\n"
            << "---------------------------------------------\n";
}

//------------------------------------------------------------------------------
// The pre-defined main function
//------------------------------------------------------------------------------

int main(int argc, char** argv) {

  PrintLogo();

  // Parse command line flags, leaving argc unchanged, but rearrange
  // the arguments in argv so that the flags are all at the beginning.
  google::ParseCommandLineFlags(&argc, &argv, false);

  if (!f2m::ValidateCommandLineFlags()) {
    LOG(ERROR) << "Validate command line failed.";
    return -1;
  }

  if (!f2m::Initialize()) {
    LOG(ERROR) << "Initialize Error.";
    return -1;
  }

  if (f2m::GetHyperParam()->is_train) {
    f2m::StartTrainWork();
  } else {
    f2m::StartPredictWork();
  }

  f2m::Finalize();

  return 0;
}
