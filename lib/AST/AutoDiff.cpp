//===-------- AutoDiff.cpp - Routines for USR generation ------------------===//
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

#include "swift/AST/AutoDiff.h"
#include "swift/Basic/LLVM.h"

using namespace swift;

SILReverseAutoDiffIndices::SILReverseAutoDiffIndices(
    unsigned source, ArrayRef<unsigned> parameters) : source(source) {
  if (parameters.empty())
    return;

  auto max = *std::max_element(parameters.begin(), parameters.end());
  this->parameters.resize(max + 1);
  int last = -1;
  for (auto paramIdx : parameters) {
    assert((int)paramIdx > last && "Parameter indices must be ascending");
    last = paramIdx;
    this->parameters.set(paramIdx);
  }
}

bool SILReverseAutoDiffIndices::operator==(
    const SILReverseAutoDiffIndices &other) const {
  if (source != other.source)
    return false;

  // The parameters are the same when they have exactly the same set bit
  // indices, even if they have different sizes.
  llvm::SmallBitVector buffer(std::max(parameters.size(),
                                       other.parameters.size()));
  buffer ^= parameters;
  buffer ^= other.parameters;
  return buffer.none();
}

/// Determines whether the type supports vector differentiation. We say that a
/// type supports vector differentiation if it conforms to `VectorNumeric` and
/// the associated type `ScalarElement` conforms to `FloatingPoint`.
static NominalTypeDecl *getAnyRealVectorTypeDecl(CanType type,
                                                 ModuleDecl *swiftModule) {
  auto &ctx = swiftModule->getASTContext();
  auto *floatingPointProtocol =
      ctx.getProtocol(KnownProtocolKind::FloatingPoint);
  auto *vectorNumericProtocol =
      ctx.getProtocol(KnownProtocolKind::VectorNumeric);
  // Look up conformance.
  auto maybeConf = swiftModule->lookupConformance(type, vectorNumericProtocol);
  if (!maybeConf)
    return nullptr;
  auto conf = *maybeConf;
  // See if the `ScalarElement` associated type conforms to `FloatingPoint`.
  DeclName scalarDeclName(ctx.getIdentifier("ScalarElement"));
  auto lookup = vectorNumericProtocol->lookupDirect(scalarDeclName);
  auto scalarAssocTy =
      cast<AssociatedTypeDecl>(lookup[0])->getDeclaredInterfaceType();
  auto scalarTy = conf.getAssociatedType(type, scalarAssocTy);
  auto scalarConf =
      swiftModule->lookupConformance(scalarTy, floatingPointProtocol);
  if (!scalarConf.hasValue())
    return nullptr;
  auto *nominal = type->getAnyNominal();
  assert(nominal && "Should've been nominal since it conforms to protocols");
  return nominal;
}

/// Determines whether the type supports scalar differentiation. We say that a
/// type supports scalar differentiation if it conforms to `FloatingPoint` and
/// the associated type `ScalarElement` conforms to `FloatingPoint`.
static NominalTypeDecl *getAnyRealScalarTypeDecl(CanType type,
                                                 ModuleDecl *swiftModule) {
  auto *fpProto = swiftModule->getASTContext()
      .getProtocol(KnownProtocolKind::FloatingPoint);
  if (!swiftModule->lookupConformance(type, fpProto))
    return nullptr;
  auto *nominal = type->getAnyNominal();
  assert(nominal && "Should've been nominal since it conforms to protocols");
  return nominal;
}

/// Determines the cotangent space of a type.
Optional<CotangentSpace> ASTContext::getCotangentSpace(CanType type) const {
  LLVM_DEBUG(getADDebugStream() << "Classifying cotangent space for "
             << type << '\n');
  auto lookup = cachedCotangentSpaces.find(type);
  if (lookup != cachedCotangentSpaces.end())
    return lookup->getSecond();
  // A helper that is used to cache the computed cotangent space for the
  // specified type and retuns the same cotangent space.
  auto cache = [&](Optional<CotangentSpace> cotangentSpace) {
    cachedCotangentSpaces.insert({type, cotangentSpace});
    return cotangentSpace;
  };
  // `Builtin.FP<...>` is a builtin real scalar space.
  if (auto *fpType = type->getAs<BuiltinFloatType>())
    return cache(CotangentSpace::getBuiltinRealScalarSpace(fpType));
  // Types that conform to `FloatingPoint` are a real scalar space.
  if (auto *nomTy = getAnyRealScalarTypeDecl(type, *this))
    return cache(CotangentSpace::getRealScalarSpace(nomTy));
  // Types that conform to `VectorNumeric` where the associated `ScalarElement`
  // conforms to `FloatingPoint` are a real vector space.
  if (auto *nomTy = getAnyRealVectorTypeDecl(type, *this))
    return cache(CotangentSpace::getRealVectorSpace(nomTy));
  // Nominal types can be either a struct or an enum.
  if (auto *nominal = type->getAnyNominal()) {
    // Fixed-layout struct types, each of whose elements has a cotangent space,
    // are a product of those cotangent spaces.
    if (auto *structDecl = dyn_cast<StructDecl>(nominal)) {
      if (structDecl->getFormalAccess() >= AccessLevel::Public &&
          !structDecl->getAttrs().hasAttribute<FixedLayoutAttr>())
        return cache(None);
      auto allMembersHaveCotangentSpace =
          llvm::all_of(structDecl->getStoredProperties(), [&](VarDecl *v) {
            return (bool)getCotangentSpace(v->getType()->getCanonicalType());
          });
      if (allMembersHaveCotangentSpace)
        return cache(CotangentSpace::getProductStruct(structDecl));
    }
    // Frozen enum types, all of whose payloads have a cotangent space, are a
    // sum of the product of payloads in each case.
    if (auto *enumDecl = dyn_cast<EnumDecl>(nominal)) {
      if (enumDecl->getFormalAccess() >= AccessLevel::Public &&
          !enumDecl->getAttrs().hasAttribute<FrozenAttr>())
        return cache(None);
      if (enumDecl->isIndirect())
        return cache(None);
      auto allMembersHaveCotangentSpace =
        llvm::all_of(enumDecl->getAllCases(), [&](EnumCaseDecl *cd) {
          return llvm::all_of(cd->getElements(), [&](EnumElementDecl *eed) {
            return llvm::all_of(*eed->getParameterList(), [&](ParamDecl *pd) {
              return (bool)
                  getCotangentSpace(pd->getType()->getCanonicalType());
            });
          });
        });
      if (allMembersHaveCotangentSpace)
        return cache(CotangentSpace::getSum(enumDecl));
    }
  }
  // Tuple types, each of whose elements has a cotangent space, are a product of
  // those cotangent space.
  if (TupleType *tupleType = type->getAs<TupleType>())
    if (llvm::all_of(tupleType->getElementTypes(), [&](Type t) {
            return (bool)getCotangentSpace(t->getCanonicalType()); }))
      return cache(CotangentSpace::getProductTuple(tupleType));
  // Otherwise, the type does not have a cotangent space. That is, it does not
  // support differentiation.
  return cache(None);
}
