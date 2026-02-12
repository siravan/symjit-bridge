# Introduction

[Symjit](https://github.com/siravan/symjit) is a lightweight just-in-time (JIT)
optimizer compiler for mathematical expressions written in Rust. It was originally
designed to compile SymPy (Python’s symbolic algebra package) expressions
into machine code and to serve as a bridge between SymPy and numerical routines
provided by NumPy and SciPy libraries.

Symjit emits AMD64 (x86-64), ARM64 (aarch64), and 64-bit RISC-V (riscv64) machine
codes on Linux, Windows, and macOS platforms. SIMD is supported on x86-64
CPUs with AVX instruction sets.

Symbolica (<https://symbolica.io/>) is a fast Rust-based Computer Algebra System.
Symbolica usually generate fast code using external compilers (e.g., using gcc to
compile synthetic c++ code). Symjit accepts Symbolica expressions and can act as
an optional code-generator for Symbolica. 

Symjit-bridge crate acts as a bridge between Symbolica and Symjit to ease generating 
JIT code for Symbolica expressions. 

# Workflow

The main workflow is using different `Runner`s. A runner corresponds to a Symbolica
`CompiledEvaluator` object. The main runners are:

* `CompiledRealRunner`, corresponding to `CompiledRealEvaluator`.
* `CompiledComplexRunner`, corresponding to `CompiledComplexEvaluator`.
* `CompiledSimdRealRunner`, corresponding to `CompiledSimdRealEvaluator`.
* `CompiledSimdComplexRunner`, corresponding to `CompiledSimdComplexEvaluator`.
* `CompiledTransposedSimdRealRunner`, similar to `CompiledSimdRealRunner` but the
    data layout is similar to `CompiledRealRunner`.
* `CompiledTransposedSimdComplexRunner`, similar to `CompiledSimdComplexRunner` but
    the data layout is similar to `CompiledComplexRunner`.
* `InterpretedRealRunner`, bytecode interpreter, generally similar to `ExpressionEvaluator`.
* `InterpretedComplexRunner`, bytecode interpreter, generally similar to `ExpressionEvaluator`.

Each runner has four main methods:

* `compile(ev: &ExpressionEvaluator<T>, config: Config)`: the main constructor. `T` is either `f64`
    or `Complex<f64>`, and `config` is an object of type `Config`. For most applications, the
    default config suffices. However, `Config.use_threads(bool)` is useful to enable multi-threading.
* `evaluate(args, outs)`: similar to the corresponding method of the `Evaluator`s.
* `save(filename)`.
* `load(filename)`.


```rust
use anyhow::Result;
use symjit_bridge::{compile, Config};

use symbolica::{
    atom::AtomCore,
    evaluate::{FunctionMap, OptimizationSettings},
    parse, symbol,
};

fn test_real_runner() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let mut runner = CompiledRealRunner::compile(&ev, Config::default())?;
    let mut outs: [f64; 1] = [0.0];
    runner.evaluate(&[3.0, 5.0], &mut outs);
    assert_eq!(outs[0], 128.0);
    Ok(())
}
```

## External Functions

Symjit has a rich set of transcendental, conditional, and logical functions (refer to
[Symjit](https://github.com/siravan/symjit) for details). It is possible to expose
these functions to Symbolica by using `add_external_function`:

```rust
fn test_external() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let mut f = FunctionMap::new();
    f.add_external_function(symbol!("sinh"), "sinh".to_string())
        .unwrap();

    let ev = parse!("sinh(x+y)")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

    let mut runner = CompiledComplexRunner::compile(&ev, Config::default())?;
    let args = [Complex::new(1.0, 2.0), Complex::new(2.0, -1.0)];
    let mut outs = [Complex::<f64>::default(); 1];
    runner.evaluate(&args, &mut outs);
    assert_eq!(outs[0], Complex::new(3.0, 1.0).sinh());
    Ok(())
}
```
