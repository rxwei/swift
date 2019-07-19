// RUN: %target-run-simple-swift
// REQUIRES: executable_test

// Test differentiation-related memory leaks.

import StdlibUnittest
import DifferentiationUnittest

var LeakCheckingTests = TestSuite("LeakChecking")

/// Execute body and check expected leak count.
func testWithLeakChecking(
  expectedLeakCount: Int = 0, file: String = #file, line: UInt = #line,
  _ body: () -> Void
) {
  // Note: compare expected leak count with relative leak count after
  // running `body`.
  // This approach is more robust than comparing leak count with zero
  // and resetting leak count to zero, which is stateful and caused bugs.
  let beforeLeakCount = _GlobalLeakCount.count
  body()
  let leakCount = _GlobalLeakCount.count - beforeLeakCount
  expectEqual(
    expectedLeakCount, leakCount, "Leaks detected: \(leakCount)",
    file: file, line: line)
}

struct ExampleLeakModel : Differentiable {
  var bias: Tracked<Float> = 2.0
  func applied(to input: Tracked<Float>) -> Tracked<Float> {
    var v = input + bias
    return v
  }
}

struct FloatPair : Differentiable & AdditiveArithmetic {
  var first, second: Tracked<Float>
  init(_ first: Tracked<Float>, _ second: Tracked<Float>) {
    self.first = first
    self.second = second
  }
}

struct Pair<T : Differentiable, U : Differentiable> : Differentiable
  where T == T.AllDifferentiableVariables, T == T.TangentVector,
        U == U.AllDifferentiableVariables, U == U.TangentVector
{
  var first: Tracked<T>
  var second: Tracked<U>
  init(_ first: Tracked<T>, _ second: Tracked<U>) {
    self.first = first
    self.second = second
  }
}

LeakCheckingTests.test("BasicLetLeakChecking") {
  testWithLeakChecking {
    let model = ExampleLeakModel()
    let x: Tracked<Float> = 1.0
    _ = model.gradient(at: x) { m, x in m.applied(to: x) }
  }

  testWithLeakChecking {
    let model = ExampleLeakModel()
    let x: Tracked<Float> = 1.0
    _ = model.gradient(at: x) { m, x in
      let (y0, y1) = (m.applied(to: x), m.applied(to: x))
      return y0 + y0 - y1
    }
  }
}

LeakCheckingTests.test("BasicVarLeakChecking") {
  testWithLeakChecking {
    var model = ExampleLeakModel()
    var x: Tracked<Float> = 1.0
    _ = gradient(at: model, x) { m, x -> Float in
      var y = x + Tracked<Float>(x.value)
      return m.applied(to: y).value
    }
  }
}

protocol DummyLayer : Differentiable {
  associatedtype Input : Differentiable
  associatedtype Output : Differentiable

  @differentiable
  func requirement(_ input: Input) -> Output
}
extension DummyLayer {
  @differentiable(vjp: vjpDefaultImpl)
  func defaultImpl(_ input: Input) -> Output {
    return requirement(input)
  }
  func vjpDefaultImpl(_ input: Input) -> (Output, (Self.Output.TangentVector) -> (Self.TangentVector, Self.Input.TangentVector)) {
    return Swift.valueWithPullback(at: self, input) { (m, i) in m.requirement(i) }
  }
}
LeakCheckingTests.test("TestProtocolDefaultDerivative") {
  struct Foo : DummyLayer {
    // NOTE: Make sure not to override `defaultImpl`.
    // To reproduce the bug, the VJP of `Foo.requirement` should dispatch to
    // `DummyLayer.vjpDefaultImpl`.

    @differentiable
    func requirement(_ input: Tracked<Float>) -> Tracked<Float> {
      return input
    }
  }

  testWithLeakChecking(expectedLeakCount: 1) {
    let x = Tracked<Float>(1)
    let model = Foo()
    _ = model.valueWithGradient { model in
      // Call the protocol default implementation method.
      model.defaultImpl(x)
    }
  }
}

LeakCheckingTests.test("NestedVarStructs") {
  func nestedstruct_var(_ x: Tracked<Float>) -> Tracked<Float> {
    var y = FloatPair(x + x, x - x)
    var z = Pair(Tracked(y), x)
    var w = FloatPair(x, x)
    y.first = w.second
    y.second = w.first
    z.first = Tracked(FloatPair(z.first.value.first - y.first,
                                z.first.value.second + y.first))
    return y.first + y.second - z.first.value.first + z.first.value.second
  }
  expectEqual((8, 2), Tracked<Float>(4).valueWithGradient(in: nestedstruct_var))
}

LeakCheckingTests.test("NestedVarTuples") {
  func nestedtuple_var(_ x: Tracked<Float>) -> Tracked<Float> {
    var y = (x + x, x - x)
    var z = (y, x)
    var w = (x, x)
    y.0 = w.1
    y.1 = w.0
    z.0.0 = z.0.0 - y.0
    z.0.1 = z.0.1 + y.0
    return y.0 + y.1 - z.0.0 + z.0.1
  }
  expectEqual((8, 2), Tracked<Float>(4).valueWithGradient(in: nestedtuple_var))
}

LeakCheckingTests.test("ClosureCaptureLeakChecking") {
  // TODO: Fix memory leak.
  testWithLeakChecking(expectedLeakCount: 1) {
    var model = ExampleLeakModel()
    let x: Tracked<Float> = 1.0

    _ = model.gradient { m in m.applied(to: x) }
    for _ in 0..<10 {
      _ = model.gradient { m in m.applied(to: x) }
    }
  }

  // TODO: Fix memory leak.
  testWithLeakChecking(expectedLeakCount: 1) {
    var model = ExampleLeakModel()
    var x: Tracked<Float> = 1.0
    _ = model.gradient { m in
      x = x + x
      var y = x + Tracked<Float>(x.value)
      return m.applied(to: y)
    }
  }

  testWithLeakChecking {
    var model = ExampleLeakModel()
    let x: Tracked<Float> = 1.0
    _ = model.gradient { m in
      var model = m
      model.bias = x
      return model.applied(to: x)
    }
  }
}

LeakCheckingTests.test("ControlFlowWithIfElse") {
  // FIXME: Fix control flow AD memory leaks.
  testWithLeakChecking(expectedLeakCount: 2) {
    var model = ExampleLeakModel()
    let x: Tracked<Float> = 1.0
    func control_flow_with_if_else(m: ExampleLeakModel, 
                                   x: Tracked<Float>) -> Tracked<Float> {
      let result: Tracked<Float>
      if x > 0 {
        result = m.applied(to: x)
      } else {
        result = x
      }
      return result
    }
    _ = model.gradient(at: x, in: control_flow_with_if_else) 
  }
}

LeakCheckingTests.test("ControlFlowWithIf") {
  // FIXME: Fix control flow AD memory leaks.
  // See related FIXME comments in adjoint value/buffer propagation in
  // lib/SILOptimizer/Mandatory/Differentiation.cpp.
  testWithLeakChecking(expectedLeakCount: 2) {
    var model = ExampleLeakModel()
    let x: Tracked<Float> = 1.0
    _ = model.gradient(at: x) { m, x in
      var result: Tracked<Float> = x
      if x > 0 {
        result = result + m.applied(to: x)
      }
      return result
    }
  }
}

LeakCheckingTests.test("ControlFlowWithIfInMethod") {
  testWithLeakChecking(expectedLeakCount: 9) {
    struct Dense : Differentiable {
      var w1: Tracked<Float>
      @noDerivative var w2: Tracked<Float>?

      func callAsFunction(_ input: Tracked<Float>) -> Tracked<Float> {
        if let w2 = w2 {
          return input * w1 * w2
        }
        return input * w1
      }
    }
    expectEqual((Dense.AllDifferentiableVariables(w1: 10), 20),
                Dense(w1: 4, w2: 5).gradient(at: 2, in: { dense, x in dense(x) }))
    expectEqual((Dense.AllDifferentiableVariables(w1: 2), 4),
                Dense(w1: 4, w2: nil).gradient(at: 2, in: { dense, x in dense(x) }))
  }
}


LeakCheckingTests.test("ControlFlowWithLoop") {
  // FIXME: Fix control flow AD memory leaks.
  // See related FIXME comments in adjoint value/buffer propagation in
  // lib/SILOptimizer/Mandatory/Differentiation.cpp.
  testWithLeakChecking(expectedLeakCount: 10) {
    func for_loop(_ x: Tracked<Float>) -> Tracked<Float> {
      var result = x
      for _ in 1..<3 {
        result = result * x
      }
      return result
    }
    expectEqual((8, 12), Tracked<Float>(2).valueWithGradient(in: for_loop))
    expectEqual((27, 27), Tracked<Float>(3).valueWithGradient(in: for_loop))
  }
}

LeakCheckingTests.test("ControlFlowWithNestedLoop") {
  // FIXME: Fix control flow AD memory leaks.
  // See related FIXME comments in adjoint value/buffer propagation in
  // lib/SILOptimizer/Mandatory/Differentiation.cpp.
  testWithLeakChecking(expectedLeakCount: 36) {
    func nested_loop(_ x: Tracked<Float>) -> Tracked<Float> {
      var outer = x
      for _ in 1..<3 {
        outer = outer * x

        var inner = outer
        var i = 1
        while i < 3 {
          inner = inner / x
          i += 1
        }
        outer = inner
      }
      return outer
    }
    expectEqual((0.5, -0.25), Tracked<Float>(2).valueWithGradient(in: nested_loop))
    expectEqual((0.25, -0.0625), Tracked<Float>(4).valueWithGradient(in: nested_loop))
  }
}

LeakCheckingTests.test("ControlFlowWithNestedTuples") {
  // FIXME: Fix control flow AD memory leaks.
  // See related FIXME comments in adjoint value/buffer propagation in
  // lib/SILOptimizer/Mandatory/Differentiation.cpp.
  testWithLeakChecking(expectedLeakCount: 29) {
    func cond_nestedtuple_var(_ x: Tracked<Float>) -> Tracked<Float> {
      // Convoluted function returning `x + x`.
      var y = (x + x, x - x)
      var z = (y, x)
      if x > 0 {
        var w = (x, x)
        y.0 = w.1
        y.1 = w.0
        z.0.0 = z.0.0 - y.0
        z.0.1 = z.0.1 + y.0
      } else {
        z = ((y.0 - x, y.1 + x), x)
      }
      return y.0 + y.1 - z.0.0 + z.0.1
    }
    expectEqual((8, 2), Tracked<Float>(4).valueWithGradient(in: cond_nestedtuple_var))
    expectEqual((-20, 2), Tracked<Float>(-10).valueWithGradient(in: cond_nestedtuple_var))
    expectEqual((-2674, 2), Tracked<Float>(-1337).valueWithGradient(in: cond_nestedtuple_var))
  }
}

LeakCheckingTests.test("ControlFlowWithNestedStructs") {
  // FIXME: Fix control flow AD memory leaks.
  // See related FIXME comments in adjoint value/buffer propagation in
  // lib/SILOptimizer/Mandatory/Differentiation.cpp.
  testWithLeakChecking(expectedLeakCount: 117) {
    func cond_nestedstruct_var(_ x: Tracked<Float>) -> Tracked<Float> {
      // Convoluted function returning `x + x`.
      var y = FloatPair(x + x, x - x)
      var z = Pair(Tracked(y), x)
      if x > 0 {
        var w = FloatPair(x, x)
        y.first = w.second
        y.second = w.first
        z.first = Tracked(FloatPair(z.first.value.first - y.first,
                                    z.first.value.second + y.first))
      } else {
        z = Pair(Tracked(FloatPair(y.first - x, y.second + x)), x)
      }
      return y.first + y.second - z.first.value.first + z.first.value.second
    }
    expectEqual((8, 2), Tracked<Float>(4).valueWithGradient(in: cond_nestedstruct_var))
    expectEqual((-20, 2), Tracked<Float>(-10).valueWithGradient(in: cond_nestedstruct_var))
    expectEqual((-2674, 2), Tracked<Float>(-1337).valueWithGradient(in: cond_nestedstruct_var))
  }
}

LeakCheckingTests.test("ControlFlowWithSwitchEnumWithPayload") {
  // FIXME: Fix control flow AD memory leaks.
  // See related FIXME comments in adjoint value/buffer propagation in
  // lib/SILOptimizer/Mandatory/Differentiation.cpp.
  testWithLeakChecking(expectedLeakCount: 47) {
    enum Enum {
      case a(Tracked<Float>)
      case b(Tracked<Float>, Tracked<Float>)
    }
    func enum_notactive2(_ e: Enum, _ x: Tracked<Float>) -> Tracked<Float> {
      var y = x
      if x > 0 {
        var z = y + y
        switch e {
        case .a: z = z - y
        case .b: y = y + x
        }
        var w = y
        if case .a = e {
          w = w + z
        }
        return w
      } else if case .b = e {
        return y + y
      }
      return x + y
    }
    expectEqual((8, 2), Tracked<Float>(4).valueWithGradient(in: { x in enum_notactive2(.a(10), x) }))
    expectEqual((20, 2), Tracked<Float>(10).valueWithGradient(in: { x in enum_notactive2(.b(4, 5), x) }))
    expectEqual((-20, 2), Tracked<Float>(-10).valueWithGradient(in: { x in enum_notactive2(.a(10), x) }))
    expectEqual((-2674, 2), Tracked<Float>(-1337).valueWithGradient(in: { x in enum_notactive2(.b(4, 5), x) }))
  }
}

runAllTests()
