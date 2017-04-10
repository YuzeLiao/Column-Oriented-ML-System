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

This file is the wrapper of mutex on both the Linux and
Windows platform.
*/

//------------------------------------------------------------------------------
// A mutex is a lockable object that is designed to signal when critical
// sections of code need exclusive access, preventing other threads with the
// same protection from executing concurrently and access the same memory
// locations.
//------------------------------------------------------------------------------

#ifndef F2M_THREAD_MUTEX_H_
#define F2M_THREAD_MUTEX_H_

#if defined _WIN32
# ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
# endif
# ifndef _WIN32_WINNT
#  define _WIN32_WINNT 0x0501  // windows xp
# endif
# define NOMINMAX 1
# include <windows.h>
#elif defined __unix__ || defined __APPLE__
# include <pthread.h>
#else
# error Unknown platform
#endif

#include <assert.h>
#include <errno.h>
#include <string.h>
#include <string>
#include <stdexcept>

#include "src/thread/scoped_locker.h"

class ConditionVariable;

#if defined _WIN32

// if _WIN32_WINNT not defined, TryEnterCriticalSection will not be declared
// in windows.h
extern "C" WINBASEAPI
BOOL WINAPI TryEnterCriticalSection(
    __inout LPCRITICAL_SECTION lpCriticalSection);

class Mutex {
 public:
  typedef ScopedLocker<Mutex> Locker;
 public:
  explicit Mutex(bool recursive = true) {
    ::InitializeCriticalSection(&m_Mutex);
  }

  ~Mutex() {
    ::DeleteCriticalSection(&m_Mutex);
  }

  void Lock() {
    ::EnterCriticalSection(&m_Mutex);
  }

  bool TryLock() {
    return ::TryEnterCriticalSection(&m_Mutex) != FALSE;
  }

  void Unlock() {
    ::LeaveCriticalSection(&m_Mutex);
  }

 private:
  // In Microsoft Windows Server 2003 Service Pack 1 and later versions of
  // Windows, the LockCount field is parsed as follows:
  // * The lowest bit shows the lock status. If this bit is 0, the critical
  //   section is locked; if it is 1, the critical section is not locked.
  // * The next bit shows whether a thread has been woken for this lock.
  //   If this bit is 0, then a thread has been woken for this lock; if it
  //   is 1, no thread has been woken.
  // * The remaining bits are the ones-complement of the number of threads
  //   waiting for the lock.
  static bool IsNewBehavior() {
    static bool result = DoIsNewBehavior();
    return result;
  }
  static bool DoIsNewBehavior() {
    Mutex mutex;
    int old_lock_count = mutex.m_Mutex.LockCount;
    mutex.Lock();
    int new_lock_count = mutex.m_Mutex.LockCount;
    return new_lock_count < old_lock_count;
  }
 private:
  Mutex(const Mutex& right);
  Mutex& operator = (const Mutex& right);
  CRITICAL_SECTION m_Mutex;
  friend class Cond;
};

#elif defined __unix__ || defined __APPLE__

class Mutex {
 public:
  typedef ScopedLocker<Mutex> Locker;
 public:
  explicit Mutex(bool recursive = true) {
    int n;
    if (recursive) {
      pthread_mutexattr_t attr;
      n = pthread_mutexattr_init(&attr);
      if (n == 0) {
        n = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        if (n == 0)
          n = pthread_mutex_init(&m_Mutex, &attr);
        n = pthread_mutexattr_destroy(&attr);
      }
    } else {
      n = pthread_mutex_init(&m_Mutex, NULL);
    }
    CheckError("Mutex::Mutex", n);
  }
  ~Mutex() {
    pthread_mutex_destroy(&m_Mutex);
  }

  void Lock() {
    CheckError("Mutex::Lock", pthread_mutex_lock(&m_Mutex));
  }

  bool TryLock() {
    int n = pthread_mutex_trylock(&m_Mutex);
    if (n == EBUSY) {
      return false;
    } else {
      CheckError("Mutex::Lock", n);
      return true;
    }
  }

  void Unlock() {
    CheckError("Mutex::Unlock", pthread_mutex_unlock(&m_Mutex));
    // NOTE: can't check unlocked here, maybe already locked by
    // other thread
  }

 private:
  static void CheckError(const char* context, int error) {
    if (error != 0) {
      std::string msg = context;
      msg += " error: ";
      msg += strerror(error);
      throw std::runtime_error(msg);
    }
  }
 private:
  Mutex(const Mutex& right);
  Mutex& operator = (const Mutex& right);
 private:
  pthread_mutex_t m_Mutex;
  friend class ConditionVariable;
};

#endif

typedef Mutex::Locker MutexLocker;

/// null mutex for template mutex param placeholder
/// NOTE: don't make this class uncopyable
class NullMutex {
 public:
  typedef ScopedLocker<NullMutex> Locker;
 public:
  NullMutex() : m_locked(false) {
  }

  void Lock() {
    m_locked = true;
  }

  bool TryLock() {
    m_locked = true;
    return true;
  }

  void Unlock() {
    m_locked = false;
  }
 private:
  bool m_locked;
};

#endif // F2M_THREAD_MUTEX_H_
