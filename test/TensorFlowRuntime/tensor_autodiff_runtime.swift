// RUN: %target-run-simple-swift
//
// TODO(SR-9110): Make this pass in dynamic compilation mode.
// %target-run-dynamic-compilation-swift
//
// REQUIRES: executable_test
// REQUIRES: swift_test_mode_optimize
//
// Tensor AD runtime tests.

import TensorFlow
import StdlibUnittest
import TensorFlowUnittest

var TensorADTests = TestSuite("TensorAD")

TensorADTests.testAllBackends("SimpleAdjointCall") {
  let adjPlus = #adjoint(Tensor<Float>.+)
  let x = Tensor<Float>(1)
  let (d0, d1) = adjPlus(x, x, x + x, x)
  expectNearlyEqual(1, d0.scalarized())
  expectNearlyEqual(1, d1.scalarized())
}

TensorADTests.testAllBackends("TestSimpleGrad") {
  func square(_ x: Tensor<Float>) -> Tensor<Float> {
    return x * x
  }
  expectEqual([0.2, 0.4, 0.6], #gradient(square)([0.1, 0.2, 0.3]))
  expectEqual([[20], [40]], #gradient(square)([[10], [20]]))
}

TensorADTests.testAllBackends("+") {
  let grad = #gradient({ (a: Tensor<Float>, b: Tensor<Float>) in (a + b).scalarized() })
  expectEqual(([1], [1]), grad([0], [0]))
  expectEqual(([1], [1]), grad([1], [10]))
}

TensorADTests.testAllBackends("-") {
  let grad = #gradient({ (a: Tensor<Float>, b: Tensor<Float>) in (a - b).scalarized() })
  expectEqual(([1], [-1]), grad([0], [0]))
  expectEqual(([1], [-1]), grad([1], [10]))
}

TensorADTests.testAllBackends("*") {
  let grad = #gradient({ (a: Tensor<Float>, b: Tensor<Float>) in (a * b).scalarized() })
  expectEqual(([0], [0]), grad([0], [0]))
  expectEqual(([10], [1]), grad([1], [10]))
}

TensorADTests.testAllBackends("/") {
  let grad = #gradient({ (a: Tensor<Float>, b: Tensor<Float>) in (a / b).scalarized() })
  expectEqual(([0.1], [-0.01]), grad([1], [10]))
}

TensorADTests.testAllBackends("•") {
  let grad = #gradient({ (a: Tensor<Float>, b: Tensor<Float>) in (a • b).scalarized() })
  expectEqual(([[0]], [[0]]), grad([[0]], [[0]]))
  expectEqual(([[10]], [[1]]), grad([[1]], [[10]]))
}

TensorADTests.testAllBackends("negate") {
  let grad = #gradient({ (a: Tensor<Float>) in -a })
  expectEqual([-1], grad([0]))
  expectEqual([-1], grad([10]))
}

runAllTests()
