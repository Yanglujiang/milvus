// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <unordered_map>
#include <vector>
#include <mutex>

#include "cnrt.h"

namespace milvus { 
namespace knowhere {

class StandardMluResources{
 public:
  StandardMluResources();

  ~StandardMluResources();

  void initMluResource(int64_t device_id);

  void freeMluResource(int64_t device_id);

  void setTempMemory(size_t size);

  void* getTempMemory();

  void freeTempMemory();

private:
  // Temp memory allocation for use with this MLU
  void* tempMemAlloc_;
  size_t tempMemAllocSize_;

};

}  // namespace knowhere
}  // namespace milvus
