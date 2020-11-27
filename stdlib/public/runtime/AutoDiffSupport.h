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

#include "swift/ABI/Metadata.h"
#include "swift/Runtime/Config.h"
#include "llvm/Support/Allocator.h"

namespace swift {

/// The header of a slot in a tape. The slot's data buffer is tail-allocated
/// after `this` and necessary alignment bytes.
struct AutoDiffTapeSlotHeader {
  /// The previous slot.
  AutoDiffTapeSlotHeader *previous;
};

/// The descriptor of a tape.
struct AutoDiffTapeDescriptor {
  /// The size of each element.
  size_t elementSize;
  /// The required alignment for each element.
  size_t elementAlignment;
  /// The allocation size for the header of a slot.  This is the size of
  /// `AutoDiffTapeSlotHeader` aligned to `elementAlignment`.
  size_t slotHeaderAllocationSize;
  /// The last slot.
  AutoDiffTapeSlotHeader *last;
};

/// A data structure responsible for allocating and walking tapes used for
/// storing linear map structures. Each basic block has a unique linear map
/// structure type, therefore the number of tapes is bound by the number of
/// basic blocks.
class AutoDiffTapeManager : public HeapObject {
private:
  llvm::BumpPtrAllocator allocator;
  llvm::SmallVector<AutoDiffTapeDescriptor, 4> tapes;

  AutoDiffTapeDescriptor &getTapeDescriptor(size_t tapeID) {
    assert(tapeID < tapes.size() && "Unrecognized tape ID");
    return tapes[tapeID];
  }

public:
  AutoDiffTapeManager();
  /// Creates a tape that stores elements of the given type and returns its tape
  /// ID.
  size_t createTape(const Metadata *elementType);
  /// Allocates a new slot on the tape with the given ID and returns a pointer
  /// to the slot's uninitialized memory.
  void *allocate(size_t tapeID);
  /// Pops a buffer from the tape with the given ID.
  void *pop(size_t tapeID);
};

/// Creates a tape manager.
SWIFT_EXPORT_FROM(swift_Differentiation) SWIFT_CC(swift)
AutoDiffTapeManager *swift_autodiff_tape_manager_create();

/// Creates a tape that stores elements of the given type in the given tape
/// manager and returns its tape ID.
SWIFT_EXPORT_FROM(swift_Differentiation) SWIFT_CC(swift)
size_t swift_autodiff_tape_create(AutoDiffTapeManager *,
                                  const Metadata *elementType);

/// Allocates a new slot on the tape with the given ID and returns a pointer
/// to the slot's uninitialized memory.
SWIFT_EXPORT_FROM(swift_Differentiation) SWIFT_CC(swift)
void *swift_autodiff_tape_allocate(AutoDiffTapeManager *, size_t tapeID);

/// Pops a buffer from the tape with the given ID.
SWIFT_EXPORT_FROM(swift_Differentiation) SWIFT_CC(swift)
void *swift_autodiff_tape_pop(AutoDiffTapeManager *, size_t tapeID);

}

#endif /* SWIFT_RUNTIME_AUTODIFF_SUPPORT_H */
