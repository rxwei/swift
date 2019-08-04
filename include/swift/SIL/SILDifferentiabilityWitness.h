//===--------------- SILDifferentiabilityWitness.h --------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2017 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
//
// This file defines the SILDifferentiabilityWitness class, which is used to
// store derivative functions.
//
//===----------------------------------------------------------------------===//

#ifndef SWIFT_SIL_SILDIFFERENTIABILITYWITNESS_H
#define SWIFT_SIL_SILDIFFERENTIABILITYWITNESS_H

#include "swift/AST/GenericSignature.h"
#include "swift/SIL/SILAllocated.h"
#include "swift/SIL/SILInstruction.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/ilist.h"

namespace swift {

class SILPrintContext;

/// A descriptor for a public property or subscript that can be resiliently
/// referenced from key paths in external modules.
class SILDifferentiabilityWitness final :
    public llvm::ilist_node<SILDifferentiabilityWitness>,
    public SILAllocated<SILDifferentiabilityWitness>
{
private:
  /// True if serialized.
  bool Serialized;
  /// True if it's a declaration.
  bool IsDeclaration;
  /// The order of differentiation.
  unsigned DifferentiationOrder;
  /// The Original function.
  SILFunction *OriginalFunction;
  /// The JVP function.
  SILFunction *JVPFunction;
  /// The VJP function.
  SILFunction *VJPFunction;
  /// The parameter indices.
  AutoDiffIndexSubset *ParameterIndices;
  /// The result indices.
  AutoDiffIndexSubset *ResultIndices;
  /// The number of generic requirements.
  unsigned NumGenericRequirements;

  Requirement *getGenericRequirementsData() {
    return reinterpret_cast<Requirement *>(this + 1);
  }

  MutableArrayRef<Requirement> getGenericRequirements() {
    return {getRequirementsData(), NumGenericRequirements};
  }

  SILDifferentiabilityWitness(bool Serialized, bool IsDeclaration,
                              unsigned DifferentiationOrder,
                              SILFunction *OriginalFunction,
                              SILFunction *JVPFunction,
                              SILFunction *VJPFunction,
                              AutoDiffIndexSubset *ParameterIndices,
                              AutoDiffIndexSubset *ResultIndices,
                              ArrayRef<Requirement> GenericRequirements)
      : Serialized(Serialized), IsDeclaration(IsDeclaration),
        DifferentiationOrder(DifferentiationOrder),
        OriginalFunction(OriginalFunction), JVPFunction(JVPFunction),
        VJPFunction(VJPFunction), ParameterIndices(ParameterIndices),
        ResultIndices(ResultIndices),
        NumGenericRequirements(GenericRequirements.size()) {
    std::uninitialized_copy(getGenericRequirements().begin(),
                            getGenericRequirements().end(),
                            GenericRequirements.begin());
  }

public:
  static SILDifferentiabilityWitness *create(
      SILModule &M, bool Serialized, bool IsDeclaration,
      unsigned DifferentiationOrder, SILFunction *OriginalFunction,
      SILFunction *JVPFunction, SILFunction *VJPFunction,
      AutoDiffIndexSubset *ParameterIndices, AutoDiffIndexSubset *ResultIndices,
      ArrayRef<Requirement> GenericRequirements);

  bool isSerialized() const { return Serialized; }
  bool isDeclaration() const { return IsDeclaration; }
  bool isDefinition() const { return !IsDeclaration; }
  unsigned getDifferentiationOrder() const { return DifferentiationOrder; }
  SILFunction *getOrigianlFunction() const { return OriginalFunction; }
  SILFunction *getJVPFunction() const { return JVPFunction; }
  SILFunction *getVJPFunction() const { return VJPFunction; }
  AutoDiffIndexSubset *getParameterIndices() const { return ParameterIndices; }
  AutoDiffIndexSubset *getResultIndices() const { return ResultIndices; }

  ArrayRef<Requirement> getGenericRequirements() const {
    return {
      const_cast<SILDifferentiabilityWitness *>(this)->getRequirementsData(),
      NumGenericRequirements};
  }

  void print(SILPrintContext &Ctx) const;
  void dump() const;

  void verify(const SILModule &M) const;
};

} // end namespace swift

namespace llvm {

//===----------------------------------------------------------------------===//
// ilist_traits for SILDifferentiabilityWitness
//===----------------------------------------------------------------------===//

template <>
struct ilist_traits<::swift::SILDifferentiabilityWitness>
    : public ilist_node_traits<::swift::SILDifferentiabilityWitness> {
  using SILDifferentiabilityWitness = ::swift::SILDifferentiabilityWitness;

public:
  static void deleteNode(SILDifferentiabilityWitness *VT) {
    VT->~SILDifferentiabilityWitness();
  }

private:
  void createNode(const SILDifferentiabilityWitness &);
};

} // namespace llvm

#endif
