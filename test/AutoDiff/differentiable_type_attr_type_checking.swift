// RUN: %target-swift-frontend -typecheck -verify %s

let _: @differentiable (Float) -> Float // okay

let _: @differentiable (String) -> Float // expected-error {{not all arguments are differentiable types; did you want to mark non-differentiable arguments as '@nodiff'?}}
let _: @differentiable (Float) -> (Float) -> Float // expected-error {{result is not a differentiable type}}
