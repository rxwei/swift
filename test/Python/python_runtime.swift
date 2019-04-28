// RUN: %target-run-simple-swift
// REQUIRES: executable_test
//
// Python runtime interop tests.

import Python
import StdlibUnittest

/// The gc module is an interface to the Python garbage collector.
let gc = Python.import("gc")
/// The tracemalloc module is a debug tool to trace memory blocks allocated by
/// Python. It is optionally imported because it is available only in Python 3.
let tracemalloc = try? Python.attemptImport("tracemalloc")

extension TestSuite {
  /// Check that running `body` does not cause Python memory leaks.
  func testWithLeakChecking(_ name: String, body: @escaping () -> Void) {
    test(name) {
      if let tracemalloc = tracemalloc {
        tracemalloc.start()
      }
      // Note: Convert to integer to prevent the integer from messing up
      // tracemalloc's count.
      let referencedObjectCount = Int(Python.len(gc.get_objects()))!
      body()
      expectEqual(referencedObjectCount, Int(Python.len(gc.get_objects()))!,
                  "Python memory leak.")
      if let tracemalloc = tracemalloc {
        expectEqual(0, Int(tracemalloc.get_traced_memory().tuple2.0)!)
        tracemalloc.stop()
      }
    }
  }
}

var PythonRuntimeTestSuite = TestSuite("PythonRuntime")
PythonLibrary.useVersion(2, 7)

PythonRuntimeTestSuite.testWithLeakChecking("CheckVersion") {
  expectEqual("2.7.", String(Python.version)!.prefix(4))
  let versionInfo = Python.versionInfo
  expectEqual(2, versionInfo.major)
  expectEqual(7, versionInfo.minor)
}

// Python.repr() in lists produces some static values that stay referenced for
// the lifetime of the program. Call here to allow leak checking to work on the
// following test.
_ = Python.repr([1])
PythonRuntimeTestSuite.testWithLeakChecking("PythonList") {
  let list: PythonObject = [0, 1, 2]
  expectEqual("[0, 1, 2]", list.description)
  expectEqual(3, Python.len(list))
  expectEqual("[0, 1, 2]", Python.str(list))
  expectEqual("<type 'list'>", Python.str(Python.type(list)))

  let polymorphicList = PythonObject(["a", 2, true, 1.5])
  expectEqual("a", polymorphicList[0])
  expectEqual(2, polymorphicList[1])
  expectEqual(true, polymorphicList[2])
  expectEqual(1.5, polymorphicList[3])
  expectEqual(1.5, polymorphicList[-1])

  polymorphicList[2] = 2
  expectEqual(2, polymorphicList[2])
}

PythonRuntimeTestSuite.testWithLeakChecking("PythonDict") {
  let dict: PythonObject = ["a": 1, 1: 0.5]
  expectEqual(2, Python.len(dict))
  expectEqual(1, dict["a"])
  expectEqual(0.5, dict[1])

  dict["b"] = "c"
  expectEqual("c", dict["b"])
  dict["b"] = "d"
  expectEqual("d", dict["b"])
}

PythonRuntimeTestSuite.testWithLeakChecking("Iterator") {
  var sum = PythonObject(0)
  for v in Python.iter([1, 2, 3]) {
    sum += v
  }
  expectEqual(6, sum)
}

PythonRuntimeTestSuite.testWithLeakChecking("Range") {
  let slice = PythonObject(5..<10)
  expectEqual(Python.slice(5, 10), slice)
  expectEqual(5, slice.start)
  expectEqual(10, slice.stop)

  let range = Range<Int>(slice)
  expectNotNil(range)
  expectEqual(5, range?.lowerBound)
  expectEqual(10, range?.upperBound)

  expectNil(Range<Int>(PythonObject(5...)))
}

PythonRuntimeTestSuite.testWithLeakChecking("PartialRangeFrom") {
  let slice = PythonObject(5...)
  expectEqual(Python.slice(5, Python.None), slice)
  expectEqual(5, slice.start)

  let range = PartialRangeFrom<Int>(slice)
  expectNotNil(range)
  expectEqual(5, range?.lowerBound)

  expectNil(PartialRangeFrom<Int>(PythonObject(..<5)))
}

PythonRuntimeTestSuite.testWithLeakChecking("PartialRangeUpTo") {
  let slice = PythonObject(..<5)
  expectEqual(Python.slice(5), slice)
  expectEqual(5, slice.stop)

  let range = PartialRangeUpTo<Int>(slice)
  expectNotNil(range)
  expectEqual(5, range?.upperBound)

  expectNil(PartialRangeUpTo<Int>(PythonObject(5...)))
}

PythonRuntimeTestSuite.testWithLeakChecking("Strideable") {
  let strideTo = stride(from: PythonObject(0), to: 100, by: 2)
  expectEqual(0, strideTo.min()!)
  expectEqual(98, strideTo.max()!)
  expectEqual([0, 2, 4, 6, 8], Array(strideTo.prefix(5)))
  expectEqual([90, 92, 94, 96, 98], Array(strideTo.suffix(5)))

  let strideThrough = stride(from: PythonObject(0), through: 100, by: 2)
  expectEqual(0, strideThrough.min()!)
  expectEqual(100, strideThrough.max()!)
  expectEqual([0, 2, 4, 6, 8], Array(strideThrough.prefix(5)))
  expectEqual([92, 94, 96, 98, 100], Array(strideThrough.suffix(5)))
}

PythonRuntimeTestSuite.testWithLeakChecking("BinaryOps") {
  expectEqual(42, PythonObject(42))
  expectEqual(42, PythonObject(2) + PythonObject(40))
  expectEqual(2, PythonObject(2) * PythonObject(3) + PythonObject(-4))

  expectEqual("abcdef", PythonObject("ab") +
                        PythonObject("cde") +
                        PythonObject("") +
                        PythonObject("f"))
  expectEqual("ababab", PythonObject("ab") * 3)

  var x = PythonObject(2)
  x += 3
  expectEqual(5, x)
  x *= 2
  expectEqual(10, x)
  x -= 3
  expectEqual(7, x)
  x /= 2
  expectEqual(3.5, x)
  x += -1
  expectEqual(2.5, x)
}

PythonRuntimeTestSuite.testWithLeakChecking("Comparable") {
  let array: [PythonObject] = [-1, 10, 1, 0, 0]
  expectEqual([-1, 0, 0, 1, 10], array.sorted())
  let list: PythonObject = [-1, 10, 1, 0, 0]
  expectEqual([-1, 0, 0, 1, 10], list.sorted())

  // Heterogeneous array/list.
  let array2: [PythonObject] = ["a", 10, "b", "b", 0]
  expectEqual([0, 10, "a", "b", "b"], array2.sorted())
  let list2: PythonObject = ["a", 10, "b", "b", 0]
  expectEqual([0, 10, "a", "b", "b"], list2.sorted())
}

PythonRuntimeTestSuite.testWithLeakChecking("Hashable") {
  func compareHashValues(_ x: PythonConvertible) {
    let a = x.pythonObject
    let b = x.pythonObject
    expectEqual(a.hashValue, b.hashValue)
  }

  compareHashValues(1)
  compareHashValues(3.14)
  compareHashValues("asdf")
  compareHashValues(PythonObject(tupleOf: 1, 2, 3))
}

PythonRuntimeTestSuite.testWithLeakChecking("RangeIteration") {
  for (index, val) in Python.range(5).enumerated() {
    expectEqual(PythonObject(index), val)
  }
}

PythonRuntimeTestSuite.testWithLeakChecking("Errors") {
  expectThrows(PythonError.exception("division by zero", traceback: nil), {
    try PythonObject(1).__truediv__.throwing.dynamicallyCall(withArguments: 0)
    // `expectThrows` does not fail if no error is thrown.
    fatalError("No error was thrown.")
  })

  expectCrash(executing: {
    let a = Python.object()
    a.foo = "bar"
  })
}

PythonRuntimeTestSuite.testWithLeakChecking("Tuple") {
  let element1: PythonObject = 0
  let element2: PythonObject = "abc"
  let element3: PythonObject = [0, 0]
  let element4: PythonObject = ["a": 0, "b": "c"]
  let pair = PythonObject(tupleOf: element1, element2)
  let (pair1, pair2) = pair.tuple2
  expectEqual(element1, pair1)
  expectEqual(element2, pair2)

  let triple = PythonObject(tupleOf: element1, element2, element3)
  let (triple1, triple2, triple3) = triple.tuple3
  expectEqual(element1, triple1)
  expectEqual(element2, triple2)
  expectEqual(element3, triple3)

  let quadruple = PythonObject(tupleOf: element1, element2, element3, element4)
  let (quadruple1, quadruple2, quadruple3, quadruple4) = quadruple.tuple4
  expectEqual(element1, quadruple1)
  expectEqual(element2, quadruple2)
  expectEqual(element3, quadruple3)
  expectEqual(element4, quadruple4)

  expectEqual(element2, quadruple[1])
}

PythonRuntimeTestSuite.testWithLeakChecking("MethodCalling") {
  let list: PythonObject = [1, 2]
  list.append(3)
  expectEqual([1, 2, 3], list)

  // Check method binding.
  let append = list.append
  append(4)
  expectEqual([1, 2, 3, 4], list)

  // Check *args/**kwargs behavior: `str.format(*args, **kwargs)`.
  let greeting: PythonObject = "{0} {first} {last}!"
  expectEqual("Hi John Smith!",
              greeting.format("Hi", first: "John", last: "Smith"))
  expectEqual("Hey Jane Doe!",
              greeting.format("Hey", first: "Jane", last: "Doe"))
}

PythonRuntimeTestSuite.testWithLeakChecking("ConvertibleFromPython") {
  // Ensure that we cover the -1 case as this is used by Python
  // to signal conversion errors.
  let minusOne: PythonObject = -1
  let zero: PythonObject = 0
  let five: PythonObject = 5
  let half: PythonObject = 0.5
  let string: PythonObject = "abc"

  expectEqual(-1, Int(minusOne))
  expectEqual(-1, Int8(minusOne))
  expectEqual(-1, Int16(minusOne))
  expectEqual(-1, Int32(minusOne))
  expectEqual(-1, Int64(minusOne))
  expectEqual(-1.0, Float(minusOne))
  expectEqual(-1.0, Double(minusOne))

  expectEqual(0, Int(zero))
  expectEqual(0.0, Double(zero))

  expectEqual(5, UInt(five))
  expectEqual(5, UInt8(five))
  expectEqual(5, UInt16(five))
  expectEqual(5, UInt32(five))
  expectEqual(5, UInt64(five))
  expectEqual(5.0, Float(five))
  expectEqual(5.0, Double(five))

  expectEqual(0.5, Float(half))
  expectEqual(0.5, Double(half))
  // Python rounds down in this case.
  expectEqual(0, Int(half))

  expectEqual("abc", String(string))

  expectNil(String(zero))
  expectNil(Int(string))
  expectNil(Double(string))
}

PythonRuntimeTestSuite.testWithLeakChecking("PythonConvertible") {
  let minusOne: PythonObject = -1
  let five: PythonObject = 5

  expectEqual(minusOne, Int(-1).pythonObject)
  expectEqual(minusOne, Int8(-1).pythonObject)
  expectEqual(minusOne, Int16(-1).pythonObject)
  expectEqual(minusOne, Int32(-1).pythonObject)
  expectEqual(minusOne, Int64(-1).pythonObject)
  expectEqual(minusOne, Float(-1).pythonObject)
  expectEqual(minusOne, Double(-1).pythonObject)

  expectEqual(five, UInt(5).pythonObject)
  expectEqual(five, UInt8(5).pythonObject)
  expectEqual(five, UInt16(5).pythonObject)
  expectEqual(five, UInt32(5).pythonObject)
  expectEqual(five, UInt64(5).pythonObject)
  expectEqual(five, Float(5).pythonObject)
  expectEqual(five, Double(5).pythonObject)
}

PythonRuntimeTestSuite.testWithLeakChecking("Optional") {
  let five: PythonObject = 5
  expectEqual(five, (5 as Int?).pythonObject)
  expectEqual(Python.None, (nil as Int?).pythonObject)

  let xx: [Int?] = [1, 2, nil, 3, nil, 4]
  let pyxx: PythonObject = [1, 2, Python.None, 3, Python.None, 4]
  expectEqual(pyxx, xx.pythonObject)
  expectEqual(xx, [Int?](pyxx))
}

PythonRuntimeTestSuite.testWithLeakChecking("SR-9230") {
  expectEqual(2, Python.len(Python.dict(a: "a", b: "b")))
}

PythonRuntimeTestSuite.testWithLeakChecking("ArrayOpsForLeakChecking") {
  expectEqual([1, 2], PythonObject([1]) + PythonObject([2]))
  expectTrue(PythonObject([1]) != PythonObject([2]))
}

// TF-78: isType() consumed refcount for type objects like `PyBool_Type`.
PythonRuntimeTestSuite.testWithLeakChecking("PythonRefCount") {
  let b: PythonObject = true
  for _ in 0...20 {
    // This triggers isType(), which used to crash after repeated invocation
    // because of reduced refcount for `PyBool_Type`.
    _ = Bool.init(b)
  }
}

PythonRuntimeTestSuite.test("ReferenceCounting") {
  // Note: gc.get_objects() only counts objects that can be part of a cycle.
  // (Like arrays and general python-objects).
  let referencedObjectCount = Python.len(gc.get_objects())
  let v = Python.list([1000, 2000, 3000])
  expectEqual(1, Python.len(gc.get_objects()) - referencedObjectCount)
}

PythonRuntimeTestSuite.test("TraceMallocReferenceCounting") {
  guard let tracemalloc = tracemalloc else { return }
  tracemalloc.start()
  expectEqual(0, Int(tracemalloc.get_traced_memory().tuple2.0)!)
  do {
    let v: PythonObject = [1, 2, 3, 4]
    expectNotEqual(0, Int(tracemalloc.get_traced_memory().tuple2.0)!)
  }
  expectEqual(0, Int(tracemalloc.get_traced_memory().tuple2.0)!)
  do {
    let v: PythonObject = 20000
    expectNotEqual(0, Int(tracemalloc.get_traced_memory().tuple2.0)!)
  }
  expectEqual(0, Int(tracemalloc.get_traced_memory().tuple2.0)!)
  do {
    let v: PythonObject = "Some String"
    expectNotEqual(0, Int(tracemalloc.get_traced_memory().tuple2.0)!)
  }
  expectEqual(0, Int(tracemalloc.get_traced_memory().tuple2.0)!)
  tracemalloc.stop()
}

runAllTests()
