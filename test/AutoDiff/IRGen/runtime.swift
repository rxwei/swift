// RUN: %target-swift-frontend -parse-stdlib %s -emit-ir | %FileCheck %s

import Swift
import _Differentiation

struct ExamplePullbackStruct<T: Differentiable> {
  var pb0: (T.TangentVector) -> T.TangentVector
}

@_silgen_name("test_tape_builtins")
func test_tape_builtins() {
  let tapeManager = Builtin.autoDiffTapeManagerCreate()
  let pbStruct = ExamplePullbackStruct<Float>(pb0: { $0 })
  let tapeID = Builtin.autoDiffTapeCreate(tapeManager, type(of: pbStruct))
  let buffer = UnsafeMutableRawPointer(Builtin.autoDiffTapeAllocate(tapeManager, tapeID))
  buffer.storeBytes(of: pbStruct, as: type(of: pbStruct))
  let poppedBuffer = Builtin.autoDiffTapePop(tapeManager, tapeID)
  let _ = UnsafeMutableRawPointer(poppedBuffer).load(as: type(of: pbStruct))
  Builtin.autoDiffTapeManagerDestroy(tapeManager)
}

// CHECK-LABEL: define{{.*}}@test_tape_builtins()
// CHECK: entry:
// CHECK:   [[TAPE_MGR:%.*]] = call swiftcc %swift.autodiff_tape_mgr* @swift_autodiff_tape_manager_create()
// CHECK:   [[TAPE_ID:%.*]] = call swiftcc {{i[0-9]+}} @swift_autodiff_tape_create(%swift.autodiff_tape_mgr* [[TAPE_MGR]], %swift.type* {{%.*}})
// CHECK:   [[BUF:%.*]] = call swiftcc i8* @swift_autodiff_tape_allocate(%swift.autodiff_tape_mgr* [[TAPE_MGR]], {{i[0-9]+}} [[TAPE_ID]])
// CHECK:   [[BUF_POPPED:%.*]] = call swiftcc i8* @swift_autodiff_tape_pop(%swift.autodiff_tape_mgr* [[TAPE_MGR]], {{i[0-9]+}} [[TAPE_ID]])
// CHECK:   call swiftcc void @swift_autodiff_tape_manager_destroy(%swift.autodiff_tape_mgr* [[TAPE_MGR]])
