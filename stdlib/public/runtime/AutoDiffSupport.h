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

/// A data structure responsible for allocating and walking tapes used for
/// storing linear map structures. Each basic block has a unique linear map
/// structure type, therefore the number of tapes is bound by the number of
/// basic blocks.
class AutoDiffTapeManager : public HeapObject {
private:
  /// The allocator.
  llvm::BumpPtrAllocator allocator;

public:
  AutoDiffTapeManager();
  /// Allocates a new slot for the given type.
  void *allocate(const Metadata *elementType);
};

/// Creates a tape manager which holds an allocator and a tail-allocated linear
/// map strucutre.
SWIFT_EXPORT_FROM(swift_Differentiation) SWIFT_CC(swift)
AutoDiffTapeManager *swift_autodiff_allocator_create(
    const Metadata *linearMapStructureType);

/// Allocates a new slot on the tape with the given ID and returns a pointer
/// to the slot's uninitialized memory.
SWIFT_EXPORT_FROM(swift_Differentiation) SWIFT_CC(swift)
void *swift_autodiff_allocator_allocate(AutoDiffTapeManager *, size_t tapeID);

}

#endif /* SWIFT_RUNTIME_AUTODIFF_SUPPORT_H */
