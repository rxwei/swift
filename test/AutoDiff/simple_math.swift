// RUN: %target-run-simple-swift
// REQUIRES: executable_test

import StdlibUnittest
#if os(macOS)
import Darwin.C
#else
import Glibc
#endif

var SimpleMathTests = TestSuite("SimpleMath")

SimpleMathTests.test("Arithmetics") {
  let foo1 = { (x: Float, y: Float) -> Float in
    return x * y
  }
  expectEqual((4, 3), gradient(at: 3, 4, in: foo1))
  let foo2 = { (x: Float, y: Float) -> Float in
    return -x * y
  }
  expectEqual((-4, -3), gradient(at: 3, 4, in: foo2))
  let foo3 = { (x: Float, y: Float) -> Float in
    return -x + y
  }
  expectEqual((-1, 1), gradient(at: 3, 4, in: foo3))
}

SimpleMathTests.test("Fanout") {
  let foo1 = { (x: Float) -> Float in
     x - x
  }
  expectEqual(0, gradient(at: 100, in: foo1))
  let foo2 = { (x: Float) -> Float in
     x + x
  }
  expectEqual(2, gradient(at: 100, in: foo2))
  let foo3 = { (x: Float, y: Float) -> Float in
    x + x + x * y
  }
  expectEqual((4, 3), gradient(at: 3, 2, in: foo3))
}

SimpleMathTests.test("FunctionCall") {
  func foo(_ x: Float, _ y: Float) -> Float {
    return 3 * x + { $0 * 3 }(3) * y
  }
  expectEqual((3, 9), gradient(at: 3, 4, in: foo))
  expectEqual(3, gradient(at: 3) { x in foo(x, 4) })
}

SimpleMathTests.test("ResultSelection") {
  func foo(_ x: Float, _ y: Float) -> (Float, Float) {
    return (x + 1, y + 2)
  }
  expectEqual((1, 0), gradient(at: 3, 3, in: { x, y in foo(x, y).0 }))
  expectEqual((0, 1), gradient(at: 3, 3, in: { x, y in foo(x, y).1 }))
}

SimpleMathTests.test("CaptureLocal") {
  let z: Float = 10
  func foo(_ x: Float) -> Float {
    return z * x
  }
  expectEqual(10, gradient(at: 0, in: foo))
}

var globalVar: Float = 10
SimpleMathTests.test("CaptureGlobal") {
  let foo: (Float) -> Float = { x in
    globalVar += 20
    return globalVar * x
  }
  expectEqual(30, gradient(at: 0, in: foo))
}

let foo: (Float) -> Float = { x in
  return x * x
}
SimpleMathTests.test("GlobalLet") {
  expectEqual(2, gradient(at: 1, in: foo))
}

var foo_diffable: @differentiable (Float) -> (Float)
  = differentiableFunction { x in (x * x, { v in 2 * x * v }) }
SimpleMathTests.test("GlobalDiffableFunc") {
  expectEqual(2, gradient(at: 1, in: foo_diffable))
  expectEqual(2, gradient(at: 1, in: { x in foo_diffable(x) }))
  expectEqual(1, gradient(at: 1, in: { (x: Float) -> Float in
    foo_diffable = { x in x + 1 };
    return foo_diffable(x)
  }))
  expectEqual(1, gradient(at: 1, in: foo_diffable))
}

SimpleMathTests.test("SideEffects") {
  func foo(x: Float) -> Float {
    var a = x
    a = a * x
    a = a * x
    return a * x
  }
  expectEqual(108, gradient(at: 3, in: foo))
}

SimpleMathTests.test("TupleSideEffects") {
  func foo(_ x: Float) -> Float {
    var tuple = (x, x)
    tuple.0 = tuple.0 * x
    tuple.0 = tuple.0 * x
    return x * tuple.0
  }
  expectEqual(27, gradient(at: 3, in: foo))

  func fooInout(_ x: Float) -> Float {
    var tuple = (x, x)
    tuple.0 *= x
    tuple.0 *= x
    return tuple.0 * tuple.0
  }
  // FIXME(TF-159): Update after activity analysis handles inout parameters.
  // expectEqual(27, gradient(at: 3, in: fooInout))
  expectEqual(0, gradient(at: 3, in: fooInout))

  func bar(_ x: Float) -> Float {
    var tuple = (x, x)
    tuple.0 = tuple.0 * x
    tuple.1 = tuple.0 * x
    return tuple.0 * tuple.1
  }
  // FIXME(TF-246): Update after zero gradient bug is fixed.
  // expectEqual(81, gradient(at: 3, in: bar))
  expectEqual(0, gradient(at: 3, in: bar))

  // FIXME(TF-201): Update after reabstraction thunks can be directly differentiated.
  /*
  func generic<T : Differentiable & AdditiveArithmetic>(_ x: T) -> T {
    var tuple = (x, x)
    tuple.0 += x
    tuple.1 += x
    return tuple.0 + tuple.0
  }
  expectEqual(1, gradient(at: 3.0, in: generic))
  */
}

// Tests TF-21.
SimpleMathTests.test("StructMemberwiseInitializer") {
  struct Foo : AdditiveArithmetic, Differentiable {
    var stored: Float
    var computed: Float {
      return stored * stored
    }
  }

  let 𝛁foo = pullback(at: Float(4), in: { input -> Foo in
    let foo = Foo(stored: input)
    return foo + foo
  })(Foo.CotangentVector(stored: 1))
  expectEqual(2, 𝛁foo)

  let 𝛁computed = gradient(at: Float(4)) { input -> Float in
    let foo = Foo(stored: input)
    return foo.computed
  }
  expectEqual(8, 𝛁computed)

  let 𝛁product = gradient(at: Float(4)) { input -> Float in
    let foo = Foo(stored: input)
    return foo.computed * foo.stored
  }
  expectEqual(16, 𝛁product)
}

runAllTests()
