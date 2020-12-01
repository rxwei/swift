//===--- AutoDiffSupport.h ------------------------------------*- C++ -*---===//
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

#ifndef SWIFT_RUNTIME_AUTODIFF_SUPPORT_H
#define SWIFT_RUNTIME_AUTODIFF_SUPPORT_H

#include "swift/Runtime/HeapObject.h"
#include "swift/Runtime/Config.h"
#include "llvm/Support/Allocator.h"

namespace swift {

class AutoDiffLinearMapContext;

/// A subcontext.
struct AutoDiffSubcontext: HeapObject {
  /// The previously allocated subcontext.
  AutoDiffSubcontext *const previous;
  /// The size of the subcontext (excluding this header).
  const size_t size;
  /// The parent context, held as a strong reference.
  AutoDiffLinearMapContext *const parentContext;

  AutoDiffSubcontext(
      AutoDiffSubcontext *const previous, size_t size,
      AutoDiffLinearMapContext *parentContext);

  static constexpr size_t getHeaderStride() {
    return llvm::alignTo<alignof(AutoDiffSubcontext)>(
        sizeof(AutoDiffSubcontext));
  }

  void *getTailMemory() {
    return reinterpret_cast<uint8_t *>(this) + getHeaderStride();
  }
};

/// A data structure responsible for efficiently allocating closure contexts for
/// linear maps such as pullbacks, including rescursive branching trace enum
/// case payloads.
class AutoDiffLinearMapContext : public HeapObject {
private:
  /// The last allocated subcontext.
  AutoDiffSubcontext *last = nullptr;
  /// The underlying allocator.
  // TODO: Use a custom allocator so that the initial slab can be tail-allocated
  // and slabs can be deallocated in a stack discipline.
  llvm::BumpPtrAllocator allocator;

public:
  /// Creates a linear map context.
  AutoDiffLinearMapContext();
  ~AutoDiffLinearMapContext();
  /// Allocates memory for a new subcontext.
  AutoDiffSubcontext *allocate(size_t size);
  /// Returns the address of the tail-allocated top-level subcontext.
  AutoDiffSubcontext *projectLastSubcontext() const;
  /// Deallocates the last allocated subcontext. The given address must be the
  /// address of the last allocated subcontext.
  void deallocate(AutoDiffSubcontext *lastSubcontext);
};

/// Creates a linear map context with a reserved capacity.
SWIFT_EXPORT_FROM(swift_Differentiation) SWIFT_CC(swift)
AutoDiffLinearMapContext *swift_autoDiffCreateLinearMapContext(
    size_t reservedCapacity);

/// Allocates memory for a new subcontext.
SWIFT_EXPORT_FROM(swift_Differentiation) SWIFT_CC(swift)
AutoDiffSubcontext *swift_autoDiffAllocateSubcontext(
    AutoDiffLinearMapContext *, size_t size);

/// Returns the address of the tail-allocated buffer in a subcontext.
SWIFT_EXPORT_FROM(swift_Differentiation) SWIFT_CC(swift)
void *swift_autoDiffProjectSubcontextBuffer(AutoDiffSubcontext *);

/// Return the previous subcontext, or null if it does not exist.
SWIFT_EXPORT_FROM(swift_Differentiation) SWIFT_CC(swift)
AutoDiffSubcontext *swift_autoDiffGetPreviousSubcontext(AutoDiffSubcontext *);

}

#endif /* SWIFT_RUNTIME_AUTODIFF_SUPPORT_H */
