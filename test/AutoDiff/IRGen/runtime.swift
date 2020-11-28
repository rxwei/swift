// RUN: %target-swift-frontend -parse-stdlib %s -emit-ir | %FileCheck %s

import Swift
import _Differentiation

struct ExamplePullbackStruct<T: Differentiable> {
  var pb0: (T.TangentVector) -> T.TangentVector
}

@_silgen_name("test_tape_builtins")
func test_tape_builtins() {
  let contextAllocator = Builtin.autoDiffContextAllocatorCreate()
  let pbStruct = ExamplePullbackStruct<Float>(pb0: { $0 })
  let rawBuffer = Builtin.autoDiffContextAllocate(contextAllocator, type(of: pbStruct))
  let buffer = UnsafeMutableRawPointer(rawBuffer)
  buffer.storeBytes(of: pbStruct, as: type(of: pbStruct))
}

// CHECK-LABEL: define{{.*}}@test_tape_builtins()
// CHECK: entry:
// CHECK:   [[ALLOCATOR:%.*]] = call swiftcc %swift.refcounted* @swift_autoDiffContextAllocatorCreate()
// CHECK:   [[BUF:%.*]] = call swiftcc i8* @swift_autoDiffContextAllocate(%swift.refcounted* [[ALLOCATOR]], %swift.type* {{%.*}})
