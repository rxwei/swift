// RUN: %target-run-eager-swift
// REQUIRES: executable_test
//
// Machine learning API AD runtime tests.
//
// NOTE: This file contains a mini high-level machine learning library. If you
// need to test other things from the `tensorflow/swift-apis` (aka.
// `DeepLearning`) library, please **copy** their definitions here instead of
// using those APIs directly. This helps avoid circular dependencies.

import TensorFlow
import StdlibUnittest
#if TPU
import TensorFlowUnittestTPU
#else
import TensorFlowUnittest
#endif

var ModelADTests = TestSuite("ModelAD")

public protocol Layer: Differentiable & KeyPathIterable
  where AllDifferentiableVariables : KeyPathIterable {
  associatedtype Input: Differentiable
  associatedtype Output: Differentiable
  @differentiable
  call func(_ input: Input) -> Output
}

@_fixed_layout
public struct Dense<Scalar: TensorFlowFloatingPoint>: Layer {
  public var weight: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  public typealias Activation =
    @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
  @noDerivative public let activation: Activation

  // FIXME(SR-9716): Remove this once the bug is fixed or worked around.
  public var allKeyPaths: [PartialKeyPath<Dense>] {
    return [\Dense.weight, \Dense.bias]
  }

  @differentiable
  public call func(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    return activation(matmul(input, weight) + bias)
  }
}

public extension Dense where Scalar.RawSignificand: FixedWidthInteger {
  init(inputSize: Int, outputSize: Int, activation: @escaping Activation) {
    self.init(weight: Tensor(glorotUniform: [inputSize, outputSize]),
              bias: Tensor(zeros: [outputSize]),
              activation: activation)
  }
}

@_fixed_layout
public struct Conv2D<Scalar: TensorFlowFloatingPoint>: Layer {
  public var filter: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
  @noDerivative public let activation: Activation
  @noDerivative public let strides: (Int, Int)
  @noDerivative public let padding: Padding

  @differentiable
  public init(
    filter: Tensor<Scalar>,
    bias: Tensor<Scalar>,
    activation: @escaping Activation,
    strides: (Int, Int),
    padding: Padding
  ) {
    self.filter = filter
    self.bias = bias
    self.activation = activation
    self.strides = strides
    self.padding = padding
  }

  @differentiable
  public call func(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
      return activation(input.convolved2D(withFilter: filter,
                                          strides: (1, strides.0, strides.1, 1),
                                          padding: padding) + bias)
  }
}

public protocol Optimizer {
  associatedtype Model: Layer
  associatedtype Scalar: FloatingPoint
  var learningRate: Scalar { get }
  mutating func update(_ variables: inout Model.AllDifferentiableVariables,
                       along vector: Model.CotangentVector)
}

public class RiemannSGD<Model: Layer, Scalar: FloatingPoint>: Optimizer
  where Model.TangentVector: VectorNumeric,
        Model.TangentVector.Scalar == Scalar {
  public var learningRate: Scalar

  public init(
    learningRate: Scalar,
    modelType: Model.Type = Model.self,
    scalarType: Scalar.Type = Scalar.self
  ) {
    self.learningRate = learningRate
  }

  public func update(_ model: inout Model.AllDifferentiableVariables,
                     along vector: Model.CotangentVector) {
    model = model.moved(
      along: learningRate * (.zero - model.tangentVector(from: vector)))
  }
}

ModelADTests.testAllBackends("SimpleLayerAD") {
  let ones = Tensor<Float>(ones: [2, 2])
  let dense = Dense<Float>(inputSize: 2, outputSize: 2, activation: { $0 })
  let grad = gradient(at: dense) { dense in dense(ones).sum() }
  expectEqual([[2, 2], [2, 2]], grad.weight)
  expectEqual([2, 2], grad.bias)
}

ModelADTests.testAllBackends("XORTraining") {
  struct Classifier: Layer {
    var l1, l2: Dense<Float>
    init(hiddenSize: Int) {
      l1 = Dense<Float>(inputSize: 2, outputSize: hiddenSize, activation: relu)
      l2 = Dense<Float>(inputSize: hiddenSize, outputSize: 1, activation: relu)
    }
    @differentiable
    call func(_ input: Tensor<Float>) -> Tensor<Float> {
      return l2(l1(input))
    }
  }
  var classifier = Classifier(hiddenSize: 4)
  let optimizer = RiemannSGD<Classifier, Float>(learningRate: 0.2)
  let x: Tensor<Float> = [[0, 0], [0, 1], [1, 0], [1, 1]]
  let y: Tensor<Float> = [0, 1, 1, 0]
  for _ in 0..<1000 {
    let 𝛁model = classifier.gradient { classifier -> Tensor<Float> in
      let ŷ = classifier(x)
      return meanSquaredError(predicted: ŷ, expected: y)
    }
    optimizer.update(&classifier.allDifferentiableVariables, along: 𝛁model)
  }
  _ = classifier([[0, 0], [0, 1], [1, 0], [1, 1]])
}

ModelADTests.testAllBackends("WithRespectToModel") {
  struct Foo<Scalar>: Differentiable where Scalar: TensorFlowFloatingPoint {
    var bar: Tensor<Scalar>
    var baz: Tensor<Scalar>

    @differentiable
    call func(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
      return bar + input
    }
  }
  let x = Tensor<Float>(0)
  var model = Foo<Float>(bar: x, baz: x)
  let 𝛁model = gradient(at: model) { model in model(x) }
  expectEqual(
    Foo<Float>.AllDifferentiableVariables(bar: Tensor(1.0), baz: Tensor(0.0)),
    𝛁model)
}

runAllTests()
