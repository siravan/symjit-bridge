use anyhow::Result;
use num_complex::Complex;
use symjit_bridge::{
    compile, CompiledComplexRunner, CompiledRealRunner, CompiledSimdComplexRunner,
    CompiledSimdRealRunner, Config,
};

use symbolica::{
    atom::AtomCore,
    evaluate::{CompileOptions, FunctionMap, OptimizationSettings},
    parse, symbol,
};

use wide::{f64x2, f64x4};

fn pass(what: &str) {
    println!("**** test {:?} passed. ****", what);
}

fn test_real() -> Result<()> {
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

fn test_complex_runner() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

    let mut runner = CompiledComplexRunner::compile(&ev, Config::default())?;

    let args = [Complex::new(2.0, 5.0), Complex::new(-2.0, 3.0)];
    let mut outs = [Complex::new(0.0, 0.0)];
    runner.evaluate(&args, &mut outs);
    assert_eq!(outs[0], Complex::new(48.0, 14.0));
    Ok(())
}

fn test_simd_real_runner() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let mut runner = CompiledSimdRealRunner::compile(&ev, Config::default())?;
    let args = [
        f64x4::new([1.0, 2.0, 3.0, 4.0]),
        f64x4::new([5.0, 4.0, 3.0, 2.0]),
    ];
    let mut outs = [f64x4::new([0.0, 0.0, 0.0, 0.0])];
    runner.evaluate(&args, &mut outs);
    assert_eq!(outs[0], f64x4::new([126.0, 66.0, 30.0, 12.0]));
    Ok(())
}

fn test_simd_complex_runner() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x - y^2")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

    let mut runner = CompiledSimdComplexRunner::compile(&ev, Config::default())?;
    let args = [
        Complex::new(
            f64x4::new([1.0, 2.0, 3.0, 4.0]),
            f64x4::new([2.0, 3.0, 4.0, 5.0]),
        ),
        Complex::new(
            f64x4::new([0.0, 1.0, 2.0, 3.0]),
            f64x4::new([-5.0, -4.0, -3.0, -2.0]),
        ),
    ];
    let mut outs = [Complex::<f64x4>::default()];
    runner.evaluate(&args, &mut outs);
    assert_eq!(
        outs[0],
        Complex::new(
            f64x4::new([26.0, 17.0, 8.0, -1.0]),
            f64x4::new([2.0, 11.0, 16.0, 17.0])
        )
    );
    Ok(())
}

pub fn main() -> Result<()> {
    test_real()?;
    pass("real");

    test_complex()?;
    pass("complex");

    test_real_runner()?;
    pass("real runner");

    test_complex_runner()?;
    pass("complex runner");

    test_simd_real_runner()?;
    pass("simd real runner");

    test_simd_complex_runner()?;
    pass("simd complex runner");

    Ok(())
}
