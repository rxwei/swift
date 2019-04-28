// RUN: %target-run-simple-swift
// REQUIRES: executable_test

import StdlibUnittest
#if os(macOS)
import Darwin.C
#else
import Glibc
#endif

var CustomDerivativesTests = TestSuite("CustomDerivatives")

// Specify non-differentiable functions.
// These will be wrapped in `differentiableFunction` and tested.

func unary(_ x: Float) -> Float {
  var x = x
  x *= 2
  return x
}

func binary(_ x: Float, _ y: Float) -> Float {
  var x = x
  x *= y
  return x
}

CustomDerivativesTests.test("differentiableFunction-unary") {
  let diffableUnary = differentiableFunction { x in
    (value: unary(x), pullback: { v in v * x * 2 })
  }
  expectEqual(20, gradient(at: 10, in: diffableUnary))
  // Test differentiation of @differentiable function.
  expectEqual(20, gradient(at: 10, in: { diffableUnary($0) }))
  expectEqual(40, gradient(at: 10, in: { diffableUnary($0) * 2 }))
}

CustomDerivativesTests.test("differentiableFunction-binary") {
  let diffableBinary = differentiableFunction { (x, y) in
    (value: binary(x, y), pullback: { v in (v * y, v * x) })
  }
  expectEqual((10, 5), gradient(at: 5, 10, in: diffableBinary))
  // Test differentiation of @differentiable function.
  expectEqual((10, 5), gradient(at: 5, 10, in: { diffableBinary($0, $1) }))
  expectEqual((20, 10), gradient(at: 5, 10, in: { diffableBinary($0, $1) * 2 }))
}

CustomDerivativesTests.test("Checkpointing") {
  var count = 0
  func cube(_ x: Float) -> Float {
    count += 1
    return x * x * x
  }
  // Test the top-level function variant of the checkpointing API.
  expectEqual(324, gradient(at: 3) { (x: Float) -> Float in
    expectEqual(0, count)
    let y = withRecomputationInPullbacks(cube)(x)
    expectEqual(1, count)
    return y * 3 * x
  })
  expectEqual(2, count)
  // Reset and test the method variant.
  count = 0
  expectEqual(324, gradient(at: 3) { (x: Float) -> Float in
    expectEqual(0, count)
    let y = x.withRecomputationInPullbacks(cube)
    expectEqual(1, count)
    return y * 3 * x
  })
  expectEqual(2, count)
}

runAllTests()
