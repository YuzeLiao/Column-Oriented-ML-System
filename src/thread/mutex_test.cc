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

This file tests mutex.h
*/

#include "gtest/gtest.h"
#include "src/thread/mutex.h"

TEST(Mutex, Lock) {
  Mutex mutex;
  mutex.Lock();
  mutex.Unlock();
}

TEST(Mutex, Locker) {
  Mutex mutex;
  {
    MutexLocker locker(&mutex);
  }
}

TEST(Mutex, LockerWithException) {
  Mutex mutex;
  try {
    MutexLocker locker(&mutex);
    throw 0;
  } catch(...) {
    // ignore ...
  }
}
