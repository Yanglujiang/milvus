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

#include <iostream>
#include "knowhere/index/vector_index/mlu/StandardMluResources.h"

namespace milvus { 
namespace knowhere {

StandardMluResources::StandardMluResources() : 
    tempMemAlloc_(nullptr),
    tempMemAllocSize_(0) {
}

StandardMluResources::~StandardMluResources() {
}

void
StandardMluResources::initMluResource(int64_t device_id) {
    cnrtInit(0);
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, device_id);
    cnrtSetCurrentDevice(dev);
}


void
StandardMluResources::setTempMemory(size_t size) {
    tempMemAllocSize_ = size;
//    CNRT_CHECK(cnrtMalloc(&tempMemAlloc_, cnrtDataTypeSize(CNRT_UINT8) * tempMemAllocSize_));
}

void*
StandardMluResources::getTempMemory() {
      return tempMemAlloc_;
}

void
StandardMluResources::freeTempMemory() {
//    CNRT_CHECK(cnrtFree(&tempMemAlloc_));
}

}  // namespace knowhere
}  // namespace milvus
