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

using namespace swift;
using namespace llvm;

AutoDiffTapeManager::AutoDiffTapeManager() {}

size_t AutoDiffTapeManager::createTape(const Metadata *elementType) {
  assert(
      elementType->getKind() == MetadataKind::Struct &&
      "Automatic differentiation tapes are for storing linear map structs, but "
      "the given type is not a struct");
  size_t index = tapes.size();
  auto *layout = elementType->getTypeLayout();
  size_t elementSize = layout->size;
  size_t elementAlignment = layout->flags.getAlignment();
  tapes.push_back({
    elementSize,
    elementAlignment,
    alignTo(sizeof(AutoDiffTapeSlotHeader), elementAlignment),
    /*last*/ nullptr
  });
  return index;
}

void *AutoDiffTapeManager::allocate(size_t tapeID) {
  auto &tapeDescriptor = getTapeDescriptor(tapeID);
  auto *slotBuffer = allocator.Allocate(
      tapeDescriptor.slotHeaderAllocationSize + tapeDescriptor.elementSize,
      tapeDescriptor.elementAlignment);
  tapeDescriptor.last =
      new (slotBuffer) AutoDiffTapeSlotHeader {tapeDescriptor.last};
  return static_cast<uint8_t *>(slotBuffer) +
      tapeDescriptor.slotHeaderAllocationSize;
}

void *AutoDiffTapeManager::pop(size_t tapeID) {
  auto &tapeDescriptor = getTapeDescriptor(tapeID);
  auto *last = tapeDescriptor.last;
  tapeDescriptor.last = last->previous;
  return reinterpret_cast<uint8_t *>(last) +
      tapeDescriptor.slotHeaderAllocationSize;
}

AutoDiffTapeManager *swift::swift_autodiff_tape_manager_create() {
  auto *buffer = (AutoDiffTapeManager *)swift_slowAlloc(
      sizeof(AutoDiffTapeManager), alignof(AutoDiffTapeManager));
  return new (buffer) AutoDiffTapeManager;
}

void swift::swift_autodiff_tape_manager_destroy(AutoDiffTapeManager *manager) {
  swift_slowDealloc(
      manager, sizeof(AutoDiffTapeManager), alignof(AutoDiffTapeManager));
}

size_t swift::swift_autodiff_tape_create(AutoDiffTapeManager *manager,
                                         const Metadata *elementType) {
  return manager->createTape(elementType);
}

void *swift::swift_autodiff_tape_allocate(AutoDiffTapeManager *manager,
                                          size_t tapeID) {
  return manager->allocate(tapeID);
}

void *swift::swift_autodiff_tape_pop(AutoDiffTapeManager *manager,
                                     size_t tapeID) {
  return manager->pop(tapeID);
}
