//===--- CanonicalizePartialApply.cpp - Canonicalize partial_apply --------===//
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

#define DEBUG_TYPE "canonicalize-partial-apply"
#include "swift/SIL/SILInstructionWorklist.h"
#include "swift/SILOptimizer/PassManager/Transforms.h"
#include "swift/SILOptimizer/Utils/SILOptFunctionBuilder.h"
#include "swift/SILOptimizer/Utils/CanonicalizeInstruction.h"
#include "swift/SILOptimizer/Utils/Differentiation/Thunk.h"

using namespace swift;

//===----------------------------------------------------------------------===//
//                              Top Level Driver
//===----------------------------------------------------------------------===//

namespace {

class CanonicalizePartialApply : public SILFunctionTransform {

public:
  CanonicalizePartialApply() {}

private:
  /// The entry point to the transformation.
  void run() override {
    bool changed = false;

    for (auto &inst : *getFunction())
      if (auto *pai = dyn_cast<PartialApplyInst>(&inst))
        changed |= canonicalize(pai);

    auto invalidation = SILAnalysis::InvalidationKind::Nothing;

    if (invalidation != SILAnalysis::InvalidationKind::Nothing)
      getPassManager()->invalidateAnalysis(getFunction(), invalidation);
  }

  bool canonicalize(PartialApplyInst *pai);
  bool canonicalizeDifferentiable(PartialApplyInst *pai);
};

bool CanonicalizePartialApply::canonicalize(PartialApplyInst *pai) {
  if (pai->getFunctionType()->isDifferentiable())
    return canonicalizeDifferentiable(pai);
  return false;
}

bool CanonicalizePartialApply::canonicalizeDifferentiable(
    PartialApplyInst *pai) {
  auto diffKind = pai->getFunctionType()->getDifferentiabilityKind();
  switch (diffKind) {
  case DifferentiabilityKind::Normal:
    SILBuilder builder(pai);
    auto orig = builder.createDifferentiableFunctionExtractOriginal(
        pai->getLoc(), pai->getCallee());
    auto jvp = builder.createDifferentiableFunctionExtract(
        pai->getLoc(), NormalDifferentiableFunctionTypeComponent::JVP,
        pai->getCallee());
    auto vjp = builder.createDifferentiableFunctionExtract(
        pai->getLoc(), NormalDifferentiableFunctionTypeComponent::VJP,
        pai->getCallee());
    SILOptFunctionBuilder fb(*this);
    SmallVector<SILValue, 8> args(pai->getArguments().begin(),
                                  pai->getArguments().end());
    auto origCurried = builder.createPartialApply(
        pai->getLoc(), orig, pai->getSubstitutionMap(), args,
        ParameterConvention::Direct_Guaranteed);

  case DifferentiabilityKind::Linear:
    llvm_unreachable("Unhandled");
  case DifferentiabilityKind::NonDifferentiable:
    llvm_unreachable("Not possible");
  }
}

} // end anonymous namespace

SILTransform *swift::createCanonicalizePartialApply() {
  return new CanonicalizePartialApply;
}
