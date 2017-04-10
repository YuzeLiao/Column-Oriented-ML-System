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
Author: Yuze Liao and Chao Ma (mctt90@gmail.com)

This files defines the MomentumUpdater class.
*/

#ifndef F2M_UPDATE_MOMENTUM_UPDATER_H_
#define F2M_UPDATE_MOMENTUM_UPDATER_H_

#include <vector>

#include "src/base/common.h"
#include "src/data/hyper_parameters.h"
#include "src/update/updater.h"

namespace f2m {

//------------------------------------------------------------------------------
// SGD has trouble navigating ravines, i.e. areas where the surface curves
// much more steeply in one dimension than in another, which are common
// around local optimal. In these scenarios, SGD oscillates across the slopes
// of the ravine while only making hesitant progress along the bottom towards
// the local optimum.
// Momentum is a method that helps accelerate SGD in the relevant direction
// and dampens oscillations. It does this by a fraction 'mu' of the update
// vector of the past time step to the current update vector:
// [ v = mu * v - learning_rate * dx ]
// [ w += v ]
// Note: some implementations exchange the signs in the equations. The
// momentum term 'mu' is usually set to 0.9 or a similar value.
//------------------------------------------------------------------------------
class MomentumUpdater : public Updater {
 public:
  // Constructor and Destructor
  MomentumUpdater() {  }
  ~MomentumUpdater() {  }

  // This function need to be invoked before update.
  void Initialize(const HyperParam& hyper_param);

  // Momentum update
  void Update(index_t key, real_t grad, Model* model);

  // Update model parameter in a mini-batch GD
  void BatchUpdate(Gradient* grad, Model* model);

  // Update a continuous model parameter
  void SeqUpdate(std::vector<real_t>& value,
                 index_t start_key,
                 Model* model);

 private:
  real_t mu_;

  DISALLOW_COPY_AND_ASSIGN(MomentumUpdater);
};

} // namespace f2m

#endif // F2M_UPDATE_MOMENTUM_UPDATER_H_
