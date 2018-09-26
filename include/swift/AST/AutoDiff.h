//===--- AutoDiff.h - Swift Automatic Differentiation ---------------------===//
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
//  SWIFT_ENABLE_TENSORFLOW
//  This file defines AST support for automatic differentiation.
//
//===----------------------------------------------------------------------===//

#ifndef SWIFT_AST_AUTODIFF_H
#define SWIFT_AST_AUTODIFF_H

#include "ASTContext.h"
#include "llvm/ADT/SmallBitVector.h"

namespace swift {

enum class AutoDiffMode {
  Forward, Reverse
};

struct AutoDiffIndexParameter {
  SourceLoc loc;
  unsigned index;
};

class AutoDiffParameter {
public:
  enum class Kind { Index, Self };

private:
  SourceLoc Loc;
  Kind Kind;
  union Value {
    struct { unsigned Index; }; // Index
    struct {};                  // Self
    Value(unsigned index) : Index(index) {}
    Value() {}
  } V;

public:
  AutoDiffParameter(SourceLoc loc, enum Kind kind, Value value)
    : Loc(loc), Kind(kind), V(value) {}

  static AutoDiffParameter getIndexParameter(SourceLoc loc, unsigned index) {
    return { loc, Kind::Index, index };
  }

  static AutoDiffParameter getSelfParameter(SourceLoc loc) {
    return { loc, Kind::Self, {} };
  }

  unsigned getIndex() const {
    assert(Kind == Kind::Index);
    return V.Index;
  }

  enum Kind getKind() const {
    return Kind;
  }

  SourceLoc getLoc() const {
    return Loc;
  }

  bool isEqual(const AutoDiffParameter &other) const {
    if (getKind() == other.getKind() && getKind() == Kind::Index)
      return getIndex() == other.getIndex();
    return getKind() == other.getKind() && getKind() == Kind::Self;
  }
};

/// SIL-level automatic differentiation indices. Consists of a source index,
/// i.e. index of the dependent result to differentiate from, and parameter
/// indices, i.e. index of independent parameters to differentiate with
/// respect to.
struct SILReverseAutoDiffIndices {
  /// The index of the dependent result to differentiate from.
  unsigned source;
  /// Indices of independent parameters to differentiate with respect to.
  llvm::SmallBitVector parameters;

  /// Creates a set of AD indices from the given source index and a bit vector
  /// representing parameter indices.
  /*implicit*/ SILReverseAutoDiffIndices(unsigned source,
                                         llvm::SmallBitVector parameters)
    : source(source), parameters(parameters) {}

  /// Creates a set of AD indices from the given source index and an array of
  /// parameter indices. Elements in `parameters` must be acending integers.
  /*implicit*/ SILReverseAutoDiffIndices(unsigned source,
                                         ArrayRef<unsigned> parameters);

  bool operator==(const SILReverseAutoDiffIndices &other) const;

  /// Queries whether the function's parameter with index `parameterIndex` is
  /// one of the parameters to differentiate with respect to.
  bool isWrtParameter(unsigned parameterIndex) const {
    return parameterIndex < parameters.size() &&
           parameters.test(parameterIndex);
  }

  void print(llvm::raw_ostream &s = llvm::outs()) const {
    s << "(source=" << source << " parameters=(";
    interleave(parameters.set_bits(),
               [&s](unsigned p) { s << p; }, [&s]{ s << ' '; });
    s << "))";
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &s,
                                     const SILReverseAutoDiffIndices &indices) {
  indices.print(s);
  return s;
}

/// Flags to define the semantics and the type signature of a gradient function.
enum class SILGradientFlags : unsigned {
  /// The gradient function is seedable, i.e. able to take a back-propagated
  /// adjoint value as the last parameter.
  Seedable = 1 << 0,

  /// The gradient function is preserving the result of the original function.
  PreservingResult = 1 << 1,

  /// The adjoint computation is "delayed". We say that the adjoint computation
  /// is delayed when when it's returned as a thunk.
  Delayed = 1 << 2
};
using SILGradientOptions = OptionSet<SILGradientFlags>;
static inline SILGradientOptions operator|(SILGradientFlags lhs,
                                           SILGradientFlags rhs) {
  return SILGradientOptions(unsigned(lhs) | unsigned(rhs));
}

/// SIL-level automatic differentiation configuration.
struct SILReverseAutoDiffConfig {
  SILReverseAutoDiffIndices indices;
  SILGradientOptions options;

  /*implicit*/
  SILReverseAutoDiffConfig(const SILReverseAutoDiffIndices &indices,
                                  SILGradientOptions options)
    : indices(indices), options(options) {}

  /*implicit*/
  SILReverseAutoDiffConfig(const SILReverseAutoDiffIndices &indices,
                                  bool seedable, bool preservingResult)
    : SILReverseAutoDiffConfig(indices, getCanonicalGradientOptions()) {}

  unsigned getSourceIndex() const {
    return indices.source;
  }

  llvm::SmallBitVector getParameterIndices() const {
    return indices.parameters;
  }

  bool isSeedable() const {
    return options.contains(SILGradientFlags::Seedable);
  }

  bool isPreservingResult() const {
    return options.contains(SILGradientFlags::PreservingResult);
  }

  bool isDelayed() const {
    return options.contains(SILGradientFlags::Delayed);
  }

  // FIXME: The master configuration should have all three gradient options
  // enabled, that is, the canonical gradient should return a delayed gradient
  // function. We need to handle this here as well as within the
  // differentiation pass.
  static SILGradientOptions getCanonicalGradientOptions() {
    return SILGradientFlags::Seedable | SILGradientFlags::PreservingResult;
  }

  /// Returns the "master" configuration, which all variants with the same
  /// parameter indices can derive from.
  static
  SILReverseAutoDiffConfig getMaster(
      const SILReverseAutoDiffIndices &indices) {
    return {
      indices,
      getCanonicalGradientOptions()
    };
  }

  SILReverseAutoDiffConfig getWithCanonicalOptions() const {
    return getMaster(indices);
  }

  bool isMaster() const {
    return options.toRaw() == getCanonicalGradientOptions().toRaw();
  }

  bool operator==(const SILReverseAutoDiffConfig &other) const {
    return indices == other.indices &&
           options.toRaw() == other.options.toRaw();
  }
};

/// A conceptual cotangent space representing the type of the adjoint.
class CotangentSpace {
public:
  /// A cotangent space kind.
  enum class Kind {
    /// `Builtin.FP<...>`.
    BuiltinRealScalar,
    /// A type that conforms to `FloatingPoint`.
    RealScalar,
    /// A type that conforms to `VectorNumeric` where the associated
    /// `ScalarElement` conforms to `FloatingPoint`.
    RealVector,
    /// A product of cotangent spaces as a struct.
    ProductStruct,
    /// A product of cotangent spaces as a tuple.
    ProductTuple,
    /// A sum of cotangent spaces.
    Sum
  };

private:
  Kind kind;
  union Value {
    // BuiltinRealScalar
    BuiltinFloatType *builtinFPType;
    // RealScalar or RealVector
    NominalTypeDecl *realNominalType;
    // ProductStruct
    StructDecl *structDecl;
    // ProductTuple
    TupleType *tupleType;
    // Sum
    EnumDecl *enumDecl;

    Value(BuiltinFloatType *builtinFP) : builtinFPType(builtinFP) {}
    Value(NominalTypeDecl *nominal) : realNominalType(nominal) {}
    Value(StructDecl *structDecl) : structDecl(structDecl) {}
    Value(TupleType *tupleType) : tupleType(tupleType) {}
    Value(EnumDecl *enumDecl) : enumDecl(enumDecl) {}
  } value;

  CotangentSpace(Kind kind, Value value)
      : kind(kind), value(value) {}

public:
  CotangentSpace() = delete;

  static CotangentSpace
  getBuiltinRealScalarSpace(BuiltinFloatType *builtinFP) {
    return {Kind::BuiltinRealScalar, builtinFP};
  }
  static CotangentSpace getRealScalarSpace(NominalTypeDecl *typeDecl) {
    return {Kind::RealScalar, typeDecl};
  }
  static CotangentSpace getRealVectorSpace(NominalTypeDecl *typeDecl) {
    return {Kind::RealVector, typeDecl};
  }
  static CotangentSpace getProductStruct(StructDecl *structDecl) {
    return {Kind::ProductStruct, structDecl};
  }
  static CotangentSpace getProductTuple(TupleType *tupleTy) {
    return {Kind::ProductTuple, tupleTy};
  }
  static CotangentSpace getSum(EnumDecl *enumDecl) {
    return {Kind::Sum, enumDecl};
  }

  bool isBuiltinRealScalarSpace() const {
    return kind == Kind::BuiltinRealScalar;
  }
  bool isRealScalarSpace() const { return kind == Kind::RealScalar; }
  bool isRealVectorSpace() const { return kind == Kind::RealVector; }
  bool isProductStruct() const { return kind == Kind::ProductStruct; }
  bool isProductTuple() const { return kind == Kind::ProductTuple; }

  Kind getKind() const { return kind; }
  BuiltinFloatType *getBuiltinRealScalarSpace() const {
    assert(kind == Kind::BuiltinRealScalar);
    return value.builtinFPType;
  }
  NominalTypeDecl *getRealScalarSpace() const {
    assert(kind == Kind::RealScalar);
    return value.realNominalType;
  }
  NominalTypeDecl *getRealVectorSpace() const {
    assert(kind == Kind::RealVector);
    return value.realNominalType;
  }
  NominalTypeDecl *getRealScalarOrVectorSpace() const {
    assert(kind == Kind::RealScalar || kind == Kind::RealVector);
    return value.realNominalType;
  }
  StructDecl *getProductStruct() const {
    assert(kind == Kind::ProductStruct);
    return value.structDecl;
  }
  TupleType *getProductTuple() const {
    assert(kind == Kind::ProductTuple);
    return value.tupleType;
  }
  EnumDecl *getSum() const {
    assert(kind == Kind::Sum);
    return value.enumDecl;
  }
};

} // end namespace swift

namespace llvm {

using swift::SILReverseAutoDiffIndices;
using swift::SILReverseAutoDiffConfig;
using swift::SILGradientFlags;
using swift::OptionSet;

template<typename T> struct DenseMapInfo;

template<> struct DenseMapInfo<SILReverseAutoDiffIndices> {
  static SILReverseAutoDiffIndices getEmptyKey() {
    return { DenseMapInfo<unsigned>::getEmptyKey(), SmallBitVector() };
  }

  static SILReverseAutoDiffIndices getTombstoneKey() {
    return { DenseMapInfo<unsigned>::getTombstoneKey(),
             SmallBitVector(sizeof(intptr_t), true) };
  }

  static unsigned getHashValue(const SILReverseAutoDiffIndices &Val) {
    auto params = Val.parameters.set_bits();
    unsigned combinedHash =
      hash_combine(~1U, DenseMapInfo<unsigned>::getHashValue(Val.source),
                   hash_combine_range(params.begin(), params.end()));
    return combinedHash;
  }

  static bool isEqual(const SILReverseAutoDiffIndices &LHS,
                      const SILReverseAutoDiffIndices &RHS) {
    return LHS == RHS;
  }
};

template<> struct DenseMapInfo<SILReverseAutoDiffConfig> {
  static SILReverseAutoDiffConfig getEmptyKey() {
    return { DenseMapInfo<SILReverseAutoDiffIndices>::getEmptyKey(), None };
  }

  static SILReverseAutoDiffConfig getTombstoneKey() {
    return {
      DenseMapInfo<SILReverseAutoDiffIndices>::getTombstoneKey(),
      SILGradientFlags::Delayed
    };
  }

  static unsigned getHashValue(const SILReverseAutoDiffConfig &Val) {
    return hash_combine(
      DenseMapInfo<SILReverseAutoDiffIndices>::getHashValue(Val.indices),
      DenseMapInfo<unsigned>::getHashValue(Val.options.toRaw())
    );
  }

  static bool isEqual(const SILReverseAutoDiffConfig &LHS,
                      const SILReverseAutoDiffConfig &RHS) {
    return DenseMapInfo<SILReverseAutoDiffIndices>
             ::isEqual(LHS.indices, RHS.indices) &&
           LHS.options.toRaw() == RHS.options.toRaw();
  }
};

} // end namespace llvm

#endif // SWIFT_AST_AUTODIFF_H
