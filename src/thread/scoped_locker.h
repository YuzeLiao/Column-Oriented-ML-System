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

This file is the implementation of ScopedLocker class.
*/

#ifndef F2M_THREAD_SCOPED_LOCKER_H_
#define F2M_THREAD_SCOPED_LOCKER_H_

#include "src/base/common.h"

//------------------------------------------------------------------------------
// ScopedLocker treat its scope as critical sections. MutexLocker take a
// mutex as the lockable object.  We can use it like this:
//
//    Mutex mutex;
//    {
//      MutexLocker locker(&mutex);
//    }
//
//    The {  } is the critical sections of code.
//------------------------------------------------------------------------------
template <typename LockType>
class ScopedLocker {
 public:
  explicit ScopedLocker(LockType* lock) : m_lock(lock) {
    m_lock->Lock();
  }
  ~ScopedLocker() {
    m_lock->Unlock();
  }
 private:
  LockType* m_lock;
};

template <typename LockType>
class ScopedReaderLocker {
 public:
  explicit ScopedReaderLocker(LockType* lock) : m_lock(lock) {
    m_lock->ReaderLock();
  }
  ~ScopedReaderLocker() {
    m_lock->ReaderUnlock();
  }
 private:
  LockType* m_lock;
};

template <typename LockType>
class ScopedWriterLocker {
 public:
  explicit ScopedWriterLocker(LockType* lock) : m_lock(*lock) {
    m_lock.WriterLock();
  }
  ~ScopedWriterLocker() {
    m_lock.WriterUnlock();
  }
 private:
  LockType& m_lock;
};

#endif // F2M_THREAD_SCOPED_LOCKER_H_
