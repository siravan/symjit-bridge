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

The main workflow is to pass a Symbolica `ExpressionEvaluator` object to 
symjit-bridge `compile` function:

```python
use anyhow::Result;
use symjit_bridge::{compile, Config};

use symbolica::{
    atom::AtomCore,
    evaluate::{FunctionMap, OptimizationSettings},
    parse, symbol,
};

fn test() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    
    let ev = parse!("x + y^2")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());
        
    let mut app = compile(&ev, Config::default())?;
    let u = app.evaluate_single(&[3.0, 4.0]);
    assert_eq!(u, 19.0);
    Ok(())
}
```

The first parameter to `compile` should be an `ExpressionEvaluator<f64>` or 
`ExpressionEvaluator<Complex<f64>>`. The second parameter to `compile` is 
a `Config` object. It is usually created by `Config::default`. It has 
multiple methods, but the more useful ones are `set_complex(bool)` 
and `set_simd(bool)`.

If successful, `compile` returns an `Application` object, which wraps
the compiled code and can be run using one of its `evaluate` functions:

* `evaluate(&mut self, args: &[T], outs: &mut [T])`, where `T` is 
    either `f64` or `Complex<f64>`.
* `evaluate_single(&mut self, args: &[T]) -> T`.
* `evaluate_simd(&mut self, args: &[S], outs: &mut [S])`, where `S` is 
    either `f64x4` for x86-64 and `f64x2` for `aarch64`.
* `evaluate_simd_single(&mut self, args: &[S]) -> S`.
* `evaluate_matrix(&mut self, args: &[f64], outs: &mut [f64], nrows: usize)`.
* `evaluate_complex_matrix(&mut self, args: &[Complex<f64>], outs: &mut [Complex<f64>], nrows: usize)`.

Note that `evaluate_matrix` and `evaluate_complex_matrix` may use SIMD instructions 
and/or multi-threading based on the status of the `Config` object passed to `compile`,
the capabilities of the processor, and `nrows`.

## Complex Expressions

```python
use num_complex::Complex;

fn test_complex() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

    let mut config = Config::default();
    config.set_complex(true);
    let mut app = compile(&ev, config)?;
    let u = app.evaluate_single(&[Complex::new(2.0, 1.0), Complex::new(-2.0, 4.0)]);
    assert_eq!(u, Complex::new(90.0, -15.0));

    Ok(())
}
```

## External Functions

Symjit has a rich set of transcendental, conditional, and logical functions (refer to 
[Symjit](https://github.com/siravan/symjit) for details). It is possible to expose 
these functions to Symbolica by using `add_external_function`:

```python
fn test_external() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let mut f = FunctionMap::new();
    f.add_external_function(symbol!("sinh"), "sinh".to_string())
        .unwrap();

    let ev = parse!("sinh(x+y)")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let mut app = compile(&ev, Config::default())?;
    let u = app.evaluate_single(&[2.0, -3.0]);
    assert!((u - f64::sinh(-1.0)).abs() < 1e-15);

    Ok(())
}
```

## Evaluating Matrices

`evaluate_matrix` and `evaluate_complex_matrix` accept matrix arguments. The 
following example shows how to use Symjit on complex matrix:

```python
fn test_complex_matrix() -> Result<()> {
    let ev = parse!("cos(x^10 + y^10)")
        .evaluator(
            &FunctionMap::new(),
            &vec![parse!("x"), parse!("y")],
            OptimizationSettings::default(),
        )
        .unwrap()
        .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

    let mut config = Config::default();
    config.set_complex(true);
    let mut app = compile(&ev, config)?;

    const N: usize = 100;
    let mut input: Vec<Complex<f64>> = vec![Complex::default(); 2 * N];
    for i in 0..N {
        input[2 * i] = Complex::new((i as f64).sin(), (i as f64).cos());
        input[2 * i + 1] = Complex::new((2.0 * i as f64).sin(), (2.0 * i as f64).cos());
    }

    let mut outs: Vec<Complex<f64>> = vec![Complex::default(); 2 * N];
    app.evaluate_complex_matrix(&input, &mut outs, N);

    assert!((outs[19].re - 1.0289805626427462).abs() < 1e-15));
    assert!((outs[19].im + 1.1072191382355374).abs() < 1e-15));

    Ok(())
}
```
