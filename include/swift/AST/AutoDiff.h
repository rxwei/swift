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

class ParsedAutoDiffParameter {
public:
  enum class Kind { Named, Self };

private:
  SourceLoc Loc;
  Kind Kind;
  union Value {
    struct { Identifier Name; }; // Index
    struct {};                  // Self
    Value(Identifier name) : Name(name) {}
    Value() {}
  } V;

public:
  ParsedAutoDiffParameter(SourceLoc loc, enum Kind kind, Value value)
    : Loc(loc), Kind(kind), V(value) {}

  static ParsedAutoDiffParameter getNamedParameter(SourceLoc loc,
                                                   Identifier name) {
    return { loc, Kind::Named, name };
  }

  static ParsedAutoDiffParameter getSelfParameter(SourceLoc loc) {
    return { loc, Kind::Self, {} };
  }

  Identifier getName() const {
    assert(Kind == Kind::Named);
    return V.Name;
  }

  enum Kind getKind() const {
    return Kind;
  }

  SourceLoc getLoc() const {
    return Loc;
  }

  bool isEqual(const ParsedAutoDiffParameter &other) const {
    if (getKind() == other.getKind() && getKind() == Kind::Named)
      return getName() == other.getName();
    return getKind() == other.getKind() && getKind() == Kind::Self;
  }
};

class AnyFunctionType;
class AutoDiffParameterIndicesBuilder;
class Type;

/// Identifies a subset of a function's parameters.
///
/// When a function is curried, identifies a subset of all parameters from all
/// parameter lists. When differentiating such functions, we treat them as fully
/// uncurried.
///
/// Works with AST-level function decls and types. Requires further lowering to
/// work with SIL-level functions and types. (In particular, tuples must be
/// exploded).
///
/// Is uniquely allocated within an ASTContext so that it can be hashed and
/// compared by opaque pointer value.
class AutoDiffParameterIndices : public llvm::FoldingSetNode {
  friend AutoDiffParameterIndicesBuilder;

public:
  /// Bits corresponding to parameters in the set are "on", and bits
  /// corresponding to parameters not in the set are "off".
  ///
  /// The bits correspond to the function's parameters in order. For example,
  ///
  ///   Function type: (A, B, C) -> R
  ///   Bits: [A][B][C]
  ///
  /// When the function is curried, the bits for the first parameter list come
  /// last. For example,
  ///
  ///   Function type: (A, B) -> (C, D) -> R
  ///   Bits: [C][D][A][B]
  ///
  /// Methods follow the same pattern:
  ///
  ///   Function type: (Self) -> (A, B, C) -> R
  ///   Bits: [A][B][C][Self]
  ///
  const llvm::SmallBitVector parameters;

  static AutoDiffParameterIndices *get(llvm::SmallBitVector parameters,
                                       ASTContext &C);

private:
  AutoDiffParameterIndices(const llvm::SmallBitVector &parameters)
      : parameters(parameters) {}

public:
  /// Allocates and initializes an `AutoDiffParameterIndices` corresponding to
  /// the given `string` generated by `getString()`. If the string is invalid,
  /// returns nullptr.
  static AutoDiffParameterIndices *create(ASTContext &C, StringRef string);

  /// Returns a textual string description of these indices,
  ///
  ///   [SU]+
  ///
  /// "S" means that the corresponding index is set
  /// "U" means that the corresponding index is unset
  std::string getString() const;

  /// Tests whether this set of parameters is empty.
  bool isEmpty() const { return parameters.none(); }

  /// Pushes the subset's parameter's types to `paramTypes`, in the order in
  /// which they appear in the function type. For example,
  ///
  ///   functionType = (A, B, C) -> R
  ///   if "A" and "C" are in the set,
  ///   ==> pushes {A, C} to `paramTypes`.
  ///
  ///   functionType = (A, B) -> (C, D) -> R
  ///   if "A", "C", and "D" are in the set,
  ///   ==> pushes {A, C, D} to `paramTypes`.
  ///
  ///   functionType = (Self) -> (A, B, C) -> R
  ///   if "Self" and "C" are in the set,
  ///   ==> pushes {Self, C} to `paramTypes`.
  ///
  void getSubsetParameterTypes(AnyFunctionType *functionType,
                               SmallVectorImpl<Type> &paramTypes) const;

  /// Returns a bitvector for the SILFunction parameters corresponding to the
  /// parameters in this set. In particular, this explodes tuples. For example,
  ///
  ///   functionType = (A, B, C) -> R
  ///   if "A" and "C" are in the set,
  ///   ==> returns 101
  ///   (because the lowered SIL type is (A, B, C) -> R)
  ///
  ///   functionType = (Self) -> (A, B, C) -> R
  ///   if "Self" and "C" are in the set,
  ///   ==> returns 0011
  ///   (because the lowered SIL type is (A, B, C, Self) -> R)
  ///
  ///   functionType = (A, (B, C), D) -> R
  ///   if "A" and "(B, C)" are in the set,
  ///   ==> returns 1110
  ///   (because the lowered SIL type is (A, B, C, D) -> R)
  ///
  llvm::SmallBitVector getLowered(AnyFunctionType *functionType) const;

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(parameters.size());
    for (unsigned setBit : parameters.set_bits())
      ID.AddInteger(setBit);
  }
};

/// Builder for `AutoDiffParameterIndices`.
class AutoDiffParameterIndicesBuilder {
  llvm::SmallBitVector parameters;

public:
  /// Start building an `AutoDiffParameterIndices` for the given function type.
  AutoDiffParameterIndicesBuilder(AnyFunctionType *functionType,
                                  bool setAllParams = false);

  /// Builds the `AutoDiffParameterIndices`, returning a pointer to an existing
  /// one if it has already been allocated in the `ASTContext`.
  AutoDiffParameterIndices *build(ASTContext &C) const;

  /// Sets the parameter at `parameterIndex`. See
  /// `AutoDiffParameterIndices::parameters` for documentation about the order.
  void setParameter(unsigned parameterIndex);

  /// Returns the number of parameters.
  unsigned size() { return parameters.size(); }
};

/// SIL-level automatic differentiation indices. Consists of a source index,
/// i.e. index of the dependent result to differentiate from, and parameter
/// indices, i.e. index of independent parameters to differentiate with
/// respect to.
///
/// When a function is curried, parameter indices can refer to parameters from
/// all parameter lists. When differentiating such functions, we treat them as
/// fully uncurried.
struct SILAutoDiffIndices {
  /// The index of the dependent result to differentiate from.
  unsigned source;
  /// Independent parameters to differentiate with respect to. The bits
  /// correspond to the function's parameters in order. For example,
  ///
  ///   Function type: (A, B, C) -> R
  ///   Bits: [A][B][C]
  ///
  /// When the function is curried, the bits for the first parameter list come
  /// last. For example,
  ///
  ///   Function type: (A, B) -> (C, D) -> R
  ///   Bits: [C][D][A][B]
  ///
  llvm::SmallBitVector parameters;

  /// Creates a set of AD indices from the given source index and a bit vector
  /// representing parameter indices.
  /*implicit*/ SILAutoDiffIndices(unsigned source,
                                  llvm::SmallBitVector parameters)
      : source(source), parameters(parameters) {}

  /// Creates a set of AD indices from the given source index and an array of
  /// parameter indices. Elements in `parameters` must be ascending integers.
  /*implicit*/ SILAutoDiffIndices(unsigned source,
                                  ArrayRef<unsigned> parameters);

  bool operator==(const SILAutoDiffIndices &other) const;

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

  std::string mangle() const {
    std::string result = "src_" + llvm::utostr(source) + "_wrt_";
    interleave(parameters.set_bits(),
               [&](unsigned idx) { result += llvm::utostr(idx); },
               [&] { result += '_'; });
    return result;
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &s,
                                     const SILAutoDiffIndices &indices) {
  indices.print(s);
  return s;
}

/// The kind of an associated function.
struct AutoDiffAssociatedFunctionKind {
  enum innerty : uint8_t {
     // The Jacobian-vector products function.
     JVP = 0,
     // The vector-Jacobian products function.
     VJP = 1
  } rawValue;

  AutoDiffAssociatedFunctionKind() = default;
  AutoDiffAssociatedFunctionKind(innerty rawValue) : rawValue(rawValue) {}
  explicit AutoDiffAssociatedFunctionKind(StringRef string);
  operator innerty() const { return rawValue; }
};

/// In conjunction with the original function decl, identifies an associated
/// autodiff function.
///
/// Is uniquely allocated within an ASTContext so that it can be hashed and
/// compared by opaque pointer value.
class AutoDiffAssociatedFunctionIdentifier : public llvm::FoldingSetNode {
  const AutoDiffAssociatedFunctionKind kind;
  const unsigned differentiationOrder;
  AutoDiffParameterIndices *const parameterIndices;

  AutoDiffAssociatedFunctionIdentifier(
      AutoDiffAssociatedFunctionKind kind, unsigned differentiationOrder,
      AutoDiffParameterIndices *parameterIndices) :
    kind(kind), differentiationOrder(differentiationOrder),
    parameterIndices(parameterIndices) {}

public:
  AutoDiffAssociatedFunctionKind getKind() const { return kind; }
  unsigned getDifferentiationOrder() const { return differentiationOrder; }
  AutoDiffParameterIndices *getParameterIndices() const {
    return parameterIndices;
  }

  static AutoDiffAssociatedFunctionIdentifier *get(
      AutoDiffAssociatedFunctionKind kind, unsigned differentiationOrder,
      AutoDiffParameterIndices *parameterIndices, ASTContext &C);

  void Profile(llvm::FoldingSetNodeID &ID) {
    ID.AddInteger(kind);
    ID.AddInteger(differentiationOrder);
    ID.AddPointer(parameterIndices);
  }
};

/// The kind of an associated type.
enum class AutoDiffAssociatedVectorSpaceKind : unsigned {
  Tangent = 0, Cotangent = 1
};

/// Automatic differentiation utility namespace.
namespace autodiff {

/// Returns the offset for an associated function at a specific differentiation
/// order.
/// This is used for both ordering in the `autodiff_function` instruction and
/// ABI layout.
///
///                Order 1       Order 2     ...
/// |----------| |-----|-----| |-----|-----| ...
/// | Original | | JVP | VJP | | JVP | VJP | ...
/// |----------| |-----|-----| |-----|-----| ...
unsigned
getOffsetForAutoDiffAssociatedFunction(unsigned order,
                                       AutoDiffAssociatedFunctionKind kind);

unsigned
getNumAutoDiffAssociatedFunctions(unsigned differentiationOrder);

// Retrieve config from the function name of a variant of
// `Builtin.autodiffApply`, e.g. `Builtin.autodiffApply_jvp_arity2_order1`.
// Returns true if the function name is parsed successfully.
bool getBuiltinAutoDiffApplyConfig(StringRef operationName,
                                   AutoDiffAssociatedFunctionKind &kind,
                                   unsigned &arity, unsigned &order,
                                   bool &rethrows, bool &isMethod);
} // end namespace autodiff

class BuiltinFloatType;
class NominalOrBoundGenericNominalType;
class TupleType;

/// A type that represents a vector space.
class VectorSpace {
public:
  /// A tangent space kind.
  enum class Kind {
    /// A type that conforms to `AdditiveArithmetic`.
    Vector,
    /// A product of vector spaces as a tuple.
    Tuple,
    /// A function type whose innermost result conforms to `AdditiveArithmetic`.
    Function
  };

private:
  Kind kind;
  union Value {
    // Vector
    Type vectorType;
    // Tuple
    TupleType *tupleType;
    // Function
    AnyFunctionType *functionType;

    Value(Type vectorType) : vectorType(vectorType) {}
    Value(TupleType *tupleType) : tupleType(tupleType) {}
    Value(AnyFunctionType *functionType) : functionType(functionType) {}
  } value;

  VectorSpace(Kind kind, Value value)
      : kind(kind), value(value) {}

public:
  VectorSpace() = delete;

  static VectorSpace getVector(Type vectorType) {
    return {Kind::Vector, vectorType};
  }
  static VectorSpace getTuple(TupleType *tupleTy) {
    return {Kind::Tuple, tupleTy};
  }
  static VectorSpace getFunction(AnyFunctionType *fnTy) {
    return {Kind::Function, fnTy};
  }

  bool isVector() const { return kind == Kind::Vector; }
  bool isTuple() const { return kind == Kind::Tuple; }

  Kind getKind() const { return kind; }
  Type getVector() const {
    assert(kind == Kind::Vector);
    return value.vectorType;
  }
  TupleType *getTuple() const {
    assert(kind == Kind::Tuple);
    return value.tupleType;
  }
  AnyFunctionType *getFunction() const {
    assert(kind == Kind::Function);
    return value.functionType;
  }

  Type getType() const;
  CanType getCanonicalType() const;
  NominalTypeDecl *getNominal() const;
};

} // end namespace swift

namespace llvm {

using swift::SILAutoDiffIndices;
using swift::OptionSet;

template<typename T> struct DenseMapInfo;

template<> struct DenseMapInfo<SILAutoDiffIndices> {
  static SILAutoDiffIndices getEmptyKey() {
    return { DenseMapInfo<unsigned>::getEmptyKey(), SmallBitVector() };
  }

  static SILAutoDiffIndices getTombstoneKey() {
    return { DenseMapInfo<unsigned>::getTombstoneKey(),
             SmallBitVector(sizeof(intptr_t), true) };
  }

  static unsigned getHashValue(const SILAutoDiffIndices &Val) {
    auto params = Val.parameters.set_bits();
    unsigned combinedHash =
      hash_combine(~1U, DenseMapInfo<unsigned>::getHashValue(Val.source),
                   hash_combine_range(params.begin(), params.end()));
    return combinedHash;
  }

  static bool isEqual(const SILAutoDiffIndices &LHS,
                      const SILAutoDiffIndices &RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

#endif // SWIFT_AST_AUTODIFF_H
