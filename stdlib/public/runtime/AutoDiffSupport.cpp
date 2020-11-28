//===--- AutoDiffSupport.cpp ----------------------------------*- C++ -*---===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2019 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "AutoDiffSupport.h"
#include "swift/Runtime/Metadata.h"
#include "swift/Runtime/HeapObject.h"

using namespace swift;
using namespace llvm;

SWIFT_CC(swift)
static void swift_autodiff_context_allocator_destroy(
    SWIFT_CONTEXT HeapObject *obj) {
  auto *manager = static_cast<AutoDiffContextAllocator *>(obj);
  swift_slowDealloc(
      manager, sizeof(AutoDiffContextAllocator),
      alignof(AutoDiffContextAllocator) - 1);
}

/// Heap metadata for an asynchronous task.
static FullMetadata<HeapMetadata> contextAllocatorHeapMetadata = {
  {
    {
      &swift_autodiff_context_allocator_destroy
    },
    {
      /*value witness table*/ nullptr
    }
  },
  {
    MetadataKind::Opaque
  }
};

AutoDiffContextAllocator::AutoDiffContextAllocator()
    : HeapObject(&contextAllocatorHeapMetadata) {}

void *AutoDiffContextAllocator::allocate(const Metadata *elementType) {
  auto *layout = elementType->getTypeLayout();
  return allocator.Allocate(layout->size, layout->flags.getAlignment());
}

AutoDiffContextAllocator *swift::swift_autoDiffContextAllocatorCreate() {
  auto *buffer = (AutoDiffContextAllocator *)swift_slowAlloc(
      sizeof(AutoDiffContextAllocator), alignof(AutoDiffContextAllocator) - 1);
  return new (buffer) AutoDiffContextAllocator;
}

void *swift::swift_autoDiffContextAllocate(
    AutoDiffContextAllocator *allocator, const Metadata *linearMapStructType) {
  return allocator->allocate(linearMapStructType);
}
