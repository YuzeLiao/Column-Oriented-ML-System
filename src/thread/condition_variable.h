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

This file defines the ConditionVariable class.
*/

#ifndef F2M_THREAD_CONDITION_VARIABLE_H_
#define F2M_THREAD_CONDITION_VARIABLE_H_

#ifndef _WIN32
#if defined __unix__ || defined __APPLE__
#include <pthread.h>
#endif
#endif

#include <assert.h>
#include "src/thread/mutex.h"

//------------------------------------------------------------------------------
// A condition variable has two operations associated with it: wait() and
// signal(). The wait() call is executed when a thread wishes to put itself
// to sleep; the signal() call is executed when a thread has changed something
// in the program and thus wants to wake a sleeping thread waiting on this
// condition.
//------------------------------------------------------------------------------
class ConditionVariable {
 public:
  ConditionVariable();
  ~ConditionVariable();

  void Signal();
  void Broadcast();

  bool Wait(Mutex* inMutex, int inTimeoutInMilSecs);
  void Wait(Mutex* inMutex);

 private:
#if defined _WIN32
  HANDLE m_hCondition;
  unsigned int m_nWaitCount;
#elif defined __unix__ || defined __APPLE__
  pthread_cond_t m_hCondition;
#endif
  static void CheckError(const char* context, int error);
};

#endif // F2M_THREAD_CONDITION_VARIABLE_H_
