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

#include "swift/Runtime/AutoDiffSupport.h"

#include "swift/ABI/Metadata.h"
#include "swift/Runtime/HeapObject.h"

using namespace swift;
using namespace llvm;

SWIFT_CC(swift)
static void destroySubcontext(SWIFT_CONTEXT HeapObject *obj) {
  auto *subcontext = static_cast<AutoDiffSubcontext *>(obj);
  subcontext->parentContext->deallocate(subcontext);
}

/// Heap metadata for a linear map context.
static FullMetadata<HeapMetadata> subcontextHeapMetadata = {
  {
    {
      &destroySubcontext
    },
    {
      /*value witness table*/ nullptr
    }
  },
  {
    MetadataKind::Opaque
  }
};

AutoDiffSubcontext::AutoDiffSubcontext(AutoDiffSubcontext *const previous,
                                       size_t size,
                                       AutoDiffLinearMapContext *parentContext)
    : HeapObject(&subcontextHeapMetadata), previous(previous), size(size),
      parentContext(parentContext) {}

SWIFT_CC(swift)
static void destroyLinearMapContext(SWIFT_CONTEXT HeapObject *obj) {
  static_cast<AutoDiffLinearMapContext *>(obj)->~AutoDiffLinearMapContext();
  free(obj);
}

/// Heap metadata for a linear map context.
static FullMetadata<HeapMetadata> linearMapContextHeapMetadata = {
  {
    {
      &destroyLinearMapContext
    },
    {
      /*value witness table*/ nullptr
    }
  },
  {
    MetadataKind::Opaque
  }
};

AutoDiffLinearMapContext::AutoDiffLinearMapContext()
    : HeapObject(&linearMapContextHeapMetadata) {
}

AutoDiffLinearMapContext::~AutoDiffLinearMapContext() {
  assert(!last && "All subcontexts should have been released");
}

AutoDiffSubcontext *AutoDiffLinearMapContext::allocate(size_t size) {
  auto *buffer = allocator.Allocate(
      AutoDiffSubcontext::getHeaderStride() + size,
      alignof(AutoDiffLinearMapContext));
  swift_retain(this); // Strongly referenced by the subcontext.
  last = new (buffer) AutoDiffSubcontext(/*previous*/ last, size, this);
  ++numAllocatedSubcontexts;
  printf("New subcontext %lx; previous last %lx; total: %zu\n", (long)last, (long)last->previous, numAllocatedSubcontexts);
  return last;
}

void AutoDiffLinearMapContext::deallocate(AutoDiffSubcontext *lastSubcontext) {
  printf("Deallocating %lx...; current last %lx\n", (long)lastSubcontext, (long)last);
  assert(last == lastSubcontext);
  last = lastSubcontext->previous;
  lastSubcontext->~AutoDiffSubcontext();
  allocator.Deallocate(
      lastSubcontext, lastSubcontext->size, alignof(AutoDiffSubcontext));
  swift_release(this);
  --numAllocatedSubcontexts;
  printf("Deallocated subcontext %lx; current last %lx; total: %zu\n", (long)lastSubcontext, (long)last, numAllocatedSubcontexts);
}

AutoDiffLinearMapContext *swift::swift_autoDiffCreateLinearMapContext(
    size_t reservedCapacity) {
  auto allocationSize = alignTo(
      sizeof(AutoDiffLinearMapContext), alignof(AutoDiffLinearMapContext))
      + reservedCapacity;
  auto *buffer = (AutoDiffLinearMapContext *)malloc(allocationSize);
  return new (buffer) AutoDiffLinearMapContext;
}

AutoDiffSubcontext *swift::swift_autoDiffAllocateSubcontext(
    AutoDiffLinearMapContext *context, size_t size) {
  return context->allocate(size);
}

void *swift::swift_autoDiffProjectSubcontextBuffer(
    AutoDiffSubcontext *subcontext) {
  printf("retain count of %lx: %d\n", (long)subcontext, (long)swift_retainCount(subcontext));
  return subcontext->getTailMemory();
}

AutoDiffSubcontext *swift::swift_autoDiffGetPreviousSubcontext(
    AutoDiffSubcontext *subcontext) {
  return subcontext->previous;
}
