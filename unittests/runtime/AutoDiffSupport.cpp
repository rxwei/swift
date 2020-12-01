//===--- AutoDiff.cpp - Automatic differentiation support runtime tests ---===//
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

#include "swift/Runtime/AutoDiffSupport.h"
#include "gtest/gtest.h"

using namespace swift;

TEST(AutoDiffLinearMapContextTest, context_lifecycle) {
  auto *ctx = swift_autoDiffCreateLinearMapContext(32);
  ASSERT_EQ(swift_retainCount(ctx), 1u);
  swift_release(ctx);
}

TEST(AutoDiffLinearMapContextTest, subcontext_linked_list) {
  struct Subcontext {
    float x, y;
  };

  auto *ctx = swift_autoDiffCreateLinearMapContext(32);
  ASSERT_EQ(swift_retainCount(ctx), 1u);
  auto *subctx1 = swift_autoDiffAllocateSubcontext(ctx, sizeof(Subcontext));
  ASSERT_EQ(swift_autoDiffGetPreviousSubcontext(subctx1), nullptr);
  ASSERT_EQ(swift_retainCount(ctx), 2u);
  auto *subctx1Buffer =
      static_cast<Subcontext *>(swift_autoDiffProjectSubcontextBuffer(subctx1));
  new (subctx1Buffer) Subcontext{1, 1};
  auto *subctx2 = swift_autoDiffAllocateSubcontext(ctx, sizeof(Subcontext));
  ASSERT_EQ(swift_autoDiffGetPreviousSubcontext(subctx2), subctx1);
  ASSERT_EQ(swift_retainCount(ctx), 3u);
  auto *subctx2Buffer =
      static_cast<Subcontext *>(swift_autoDiffProjectSubcontextBuffer(subctx2));
  new (subctx2Buffer) Subcontext{2, 2};
  swift_release(subctx2);
  ASSERT_EQ(swift_retainCount(ctx), 2u);
  swift_release(subctx1);
  ASSERT_EQ(swift_retainCount(ctx), 1u);
  swift_release(ctx);
}
