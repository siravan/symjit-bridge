use anyhow::{anyhow, Result};
use std::thread;

// use numerica::domains::float::Complex;

use symjit_bridge::{
    compile, CompiledComplexRunner, CompiledRealRunner, Complex, ComplexFloat, Config, Defuns,
    InterpretedComplexRunner, InterpretedRealRunner,
};

use symjit::Applet;

use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        float,
        integer::IntegerRing,
        rational::{Fraction, Rational},
    },
    evaluate::{ExpressionEvaluator, FunctionMap, OptimizationSettings},
    parse, symbol, try_parse,
};

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use rand::prelude::*;
use wide::f64x4;

type ExternalFunction<T> = Box<dyn Fn(&[T]) -> T + Send + Sync>;

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

    let app = compile(&ev, Config::default(), 0)?;
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
    let app = compile(&ev, config, 0)?;
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

    let runner = CompiledRealRunner::compile(&ev, Config::default())?;
    let app = runner.seal()?;
    let mut outs: [f64; 1] = [0.0];
    app.evaluate(&[3.0, 5.0], &mut outs);
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

    let runner = CompiledComplexRunner::compile(&ev, Config::default())?;
    let app = runner.seal()?;

    let args = [Complex::new(2.0, 5.0), Complex::new(-2.0, 3.0)];
    let mut outs = [Complex::new(0.0, 0.0)];
    app.evaluate(&args, &mut outs);
    assert_eq!(outs[0], Complex::new(48.0, 14.0));
    Ok(())
}

fn test_scattered_simd_real_runner() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let runner = CompiledRealRunner::compile(&ev, Config::default())?;
    let app = runner.seal()?;

    let args: Vec<f64> = (0..8).map(|x| f64::from(x)).collect();
    let mut outs = [0.0; 4];
    app.evaluate_matrix(&args, &mut outs, 4);
    assert_eq!(&outs, &[1.0, 29.0, 129.0, 349.0]);
    Ok(())
}

fn test_scattered_simd_complex_runner() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x + y^2")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

    const N: usize = 97;

    let app = CompiledComplexRunner::compile(&ev, Config::default())?.seal()?;
    let args: Vec<Complex<f64>> = (0..N * 2)
        .map(|x| Complex::new(f64::from(x as i32), -1.0))
        .collect();
    let mut outs = [Complex::<f64>::default(); N];
    app.evaluate_matrix(&args, &mut outs, N);

    let mut res: Vec<Complex<f64>> = Vec::new();
    for i in 0..4 {
        let x = Complex::new((2 * i) as f64, -1.0);
        let y = Complex::new((2 * i + 1) as f64, -1.0);
        res.push(x + y * y);
    }

    for i in 0..4 {
        assert_eq!(outs[i], res[i]);
    }

    Ok(())
}

fn test_interpreted_real_runner() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let mut runner = InterpretedRealRunner::compile(&ev, Config::default())?;
    let mut outs: [f64; 1] = [0.0];

    runner.evaluate(&[3.0, 5.0], &mut outs);
    assert_eq!(outs[0], 128.0);
    Ok(())
}

fn test_interpreted_complex_runner() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

    let mut runner = InterpretedComplexRunner::compile(&ev, Config::default())?;

    let args = [Complex::new(2.0, 5.0), Complex::new(-2.0, 3.0)];
    let mut outs = [Complex::new(0.0, 0.0)];
    runner.evaluate(&args, &mut outs);
    assert_eq!(outs[0], Complex::new(48.0, 14.0));
    Ok(())
}

fn test_external() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let mut f = FunctionMap::new();
    f.add_external_function(symbol!("sinh"), "sinh".to_string())
        .unwrap();

    let ev = parse!("sinh(x+y)")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

    let runner = CompiledComplexRunner::compile(&ev, Config::default())?;
    let args = [Complex::new(1.0, 2.0), Complex::new(2.0, -1.0)];
    let mut outs = [Complex::<f64>::default(); 1];
    runner.evaluate(&args, &mut outs);
    assert_eq!(outs[0], Complex::new(3.0, 1.0).sinh());
    Ok(())
}

fn test_external_save() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let mut f = FunctionMap::new();
    f.add_external_function(symbol!("test"), "test".to_string())
        .unwrap();

    let ev = parse!("test(x, y, -(x + y))")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let mut df = Defuns::new();
    let f: ExternalFunction<f64> = Box::new(|x: &[f64]| x.iter().sum::<f64>());
    df.add_sliced_func("test", f)?;

    let app = CompiledRealRunner::compile_with_funcs(&ev, Config::from_defuns(df)?, 0)?;
    app.save("test_external.sjb")?;

    Ok(())
}

fn test_external_load() -> Result<()> {
    let mut f = FunctionMap::<f64>::new();
    f.add_external_function(symbol!("test"), "test".to_string())
        .unwrap();

    let mut df = Defuns::new();
    let f: ExternalFunction<f64> = Box::new(|x: &[f64]| x.iter().sum::<f64>());
    df.add_sliced_func("test", f)?;

    let config = Config::from_defuns(df)?;
    let applet = CompiledRealRunner::load("test_external.sjb", &config)?.seal()?;

    const N: usize = 1;
    let args: Vec<f64> = (0..N * 2).map(|x| x as f64).collect();
    let mut outs: Vec<f64> = vec![0.0; N];
    applet.evaluate(&args, &mut outs);

    for i in 0..N {
        assert!(f64::abs(outs[i]) < 1e-15);
    }

    Ok(())
}

fn test_external_func_bytecode() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let mut f = FunctionMap::new();
    f.add_external_function(symbol!("test"), "test".to_string())
        .unwrap();

    let ev = parse!("test(x, y, 1.0/(x * y))")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let mut df = Defuns::new();
    let f: ExternalFunction<f64> = Box::new(|x: &[f64]| x.iter().product::<f64>());
    df.add_sliced_func("test", f)?;

    let mut config = Config::from_name("bytecode", Config::default().opt)?;
    config.set_defuns(df);
    let runner = CompiledRealRunner::compile_with_funcs(&ev, config, 0)?;

    // runner.app.dump("test.bin", "scalar");

    const N: usize = 77;
    let mut rng = rand::rng();
    let args: Vec<f64> = (0..N * 2).map(|_| rng.random::<f64>()).collect();
    let mut outs: Vec<f64> = vec![0.0; N];
    runner.evaluate(&args, &mut outs);

    for i in 0..N {
        assert!(f64::abs(outs[i] - 1.0) < 1e-14);
    }

    Ok(())
}

fn test_external_func_complex() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let mut f = FunctionMap::new();
    f.add_external_function(symbol!("test"), "test".to_string())
        .unwrap();

    let ev = parse!("test(x, y, -(x + y))")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

    let mut rng = rand::rng();

    let mut df = Defuns::new();
    let f: ExternalFunction<Complex<f64>> =
        Box::new(|x: &[Complex<f64>]| Complex::new(1.0, 0.0) + x.iter().sum::<Complex<f64>>());
    df.add_sliced_func("test", f)?;

    let mut config = Config::from_defuns(df)?;
    config.set_simd(true);
    let runner = CompiledComplexRunner::compile_with_funcs(&ev, config, 0)?;

    // runner.app.dump("test.bin", "simd");

    const N: usize = 2;

    let args: Vec<Complex<f64>> = (0..N * 2)
        .map(|_| Complex::new(rng.random(), rng.random()))
        .collect();
    let mut outs: Vec<Complex<f64>> = vec![Complex::default(); N];

    runner.app.evaluate_matrix(&args, &mut outs, N);

    for i in 0..N {
        assert!((outs[i] - Complex::new(1.0, 0.0)).abs() < 1e-14);
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn test_external_simd_func() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let mut f = FunctionMap::new();
    f.add_external_function(symbol!("test"), "test".to_string())
        .unwrap();

    let ev = parse!("test(x, y + sin(x)^2, cos(x)^2 - x)")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let mut df = Defuns::new();
    let f: ExternalFunction<f64x4> = Box::new(|x: &[f64x4]| x.iter().sum());
    df.add_sliced_func("test", f)?;

    let config = Config::from_defuns(df)?;
    let app = CompiledRealRunner::compile_with_funcs(&ev, config, 0)?.seal()?;

    const N: usize = 131;
    let mut args = vec![f64x4::default(); 2 * N];

    for i in 0..N {
        let x = i as f64;
        args[i * 2] = f64x4::from([x, 2.0 * x, -x, 3.0]);
        args[i * 2 + 1] = f64x4::from(-1.0);
    }

    let mut outs: Vec<f64x4> = vec![f64x4::from(0.0); N];
    app.evaluate(&args, &mut outs);

    for i in 0..N {
        let a = outs[i].as_array();
        assert!(a.iter().sum::<f64>().abs() < 1e-13);
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn test_external_simd_complex_func() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let mut f = FunctionMap::new();
    f.add_external_function(symbol!("test"), "test".to_string())
        .unwrap();

    let ev = parse!("test(x, y, -(x + y))")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

    let mut df = Defuns::new();
    let f: ExternalFunction<Complex<f64x4>> = Box::new(|x: &[Complex<f64x4>]| {
        Complex::new(x[0].re + x[1].re + x[2].re, x[0].im + x[1].im + x[2].im)
    });
    df.add_sliced_func("test", f)?;

    let config = Config::from_defuns(df)?;
    let app = CompiledComplexRunner::compile_with_funcs(&ev, config, 0)?.seal()?;

    const N: usize = 135;
    let args: Vec<Complex<f64x4>> = (0..N * 2)
        .map(|x| Complex::new(f64x4::from(x as f64), f64x4::from((x + 1) as f64)))
        .collect();
    let mut outs: Vec<Complex<f64x4>> = vec![Complex::new(f64x4::default(), f64x4::default()); N];
    app.evaluate(&args, &mut outs);

    for i in 0..N {
        let a = outs[i].re.as_array();
        assert!(a[0].abs() < 1e-14);
    }

    Ok(())
}

fn test_string_real() -> Result<()> {
    const N: usize = 97;

    /*
    *  test_instructions.txt is generated as:
    *
       s = str(E("if(x, 3*sin(y)^2, 3*sin(z)^2) + if(x, 3*cos(y)^2, 3*cos(z)^2)")
           .evaluator({}, {}, [S("x"), S("y"), S("z")],
           conditionals=[S("if")]).get_instructions())
    */
    let model = std::fs::read_to_string("test_instructions.txt")?;
    let app = CompiledRealRunner::compile_string(model, Config::default())?.seal()?;

    let mut args: Vec<f64> = vec![0.0; N * 3];
    let mut rng = rand::rng();

    for i in 0..N {
        args[i * 3] = if rng.random::<f64>() < 0.5 { 0.0 } else { 1.0 };
        args[i * 3 + 1] = rng.random::<f64>();
        args[i * 3 + 2] = rng.random::<f64>();
    }

    let mut outs = [0.0; N];
    app.evaluate_matrix(&args, &mut outs, N);

    for i in 0..N {
        let delta = outs[i] - 3.0;
        assert!(delta.abs() < 1e-15);
    }

    Ok(())
}

fn test_string_complex() -> Result<()> {
    const N: usize = 97;

    /*
    *  test_instructions.txt is generated as:
    *
       s = str(E("if(x, 3*sin(y)^2, 3*sin(z)^2) + if(x, 3*cos(y)^2, 3*cos(z)^2)")
           .evaluator({}, {}, [S("x"), S("y"), S("z")],
           conditionals=[S("if")]).get_instructions())
    */
    let model = std::fs::read_to_string("test_instructions.txt")?;
    let app = CompiledComplexRunner::compile_string(model, Config::default())?.seal()?;

    let mut args: Vec<Complex<f64>> = vec![Complex::default(); N * 3];
    let mut rng = rand::rng();

    for i in 0..N {
        args[i * 3] = if rng.random::<f64>() < 0.5 {
            Complex::default()
        } else {
            Complex::new(1.0, 0.0)
        };
        args[i * 3 + 1] = Complex::new(rng.random::<f64>(), rng.random::<f64>());
        args[i * 3 + 2] = Complex::new(rng.random::<f64>(), rng.random::<f64>());
    }

    let mut outs = [Complex::default(); N];
    app.evaluate_matrix(&args, &mut outs, N);

    for i in 0..N {
        let delta = outs[i] - 3.0;
        assert!(delta.abs() < 1e-14);
    }

    Ok(())
}

fn run(app: Applet, x: f64) {
    let mut outs: [f64; 1] = [0.0];
    app.evaluate(&[3.0 + x, 5.0 + x], &mut outs);
    let y = (3.0 + x) + (5.0 + x) * (5.0 + x) * (5.0 + x);
    println!("from a thread {} vs {}", outs[0], y);
}

fn test_threads_runner() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let app = CompiledRealRunner::compile(&ev, Config::default())?.seal()?;
    let mut handles = vec![];

    for i in 0..10 {
        let a = app.clone();
        handles.push(thread::spawn(move || run(a, i as f64)));
    }

    for h in handles {
        h.join().unwrap();
    }

    Ok(())
}

fn run_application(app: Arc<Applet>, x: f64) {
    let mut outs: [f64; 1] = [0.0];
    app.evaluate(&[3.0 + x, 5.0 + x], &mut outs);
    let y = (3.0 + x) + (5.0 + x) * (5.0 + x) * (5.0 + x);
    println!("from a thread {} vs {}", outs[0], y);
}

use std::sync::Arc;

fn test_threads_application() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let runner = CompiledRealRunner::compile(&ev, Config::default())?.seal()?;
    let app = Arc::new(runner);
    //let app = Arc::new(runner.app.as_applet().clone());
    let mut handles = vec![];

    for i in 0..10 {
        let a = app.clone();
        handles.push(thread::spawn(move || run_application(a, i as f64)));
    }

    for h in handles {
        h.join().unwrap();
    }

    Ok(())
}

/* ************************************************** */

const V_VALUE: &str = "1286387037723327/2500000000000";
const P_VALUE: &str = "884279719003555/281474976710656";
const PARAM_NAME: &str = "o";
const PARAM_VALUE: Complex<f64> = Complex { re: 1.0, im: 0.0 };

fn load_expression(path: &Path) -> String {
    fs::read_to_string(path).unwrap()
}

fn parse_complex_rational(src: &str) -> Result<float::Complex<Rational>> {
    let atom = try_parse!(src).map_err(|e| anyhow!(e))?;
    float::Complex::<Rational>::try_from(atom.as_view()).map_err(|e| anyhow!(e))
}

fn build_evaluator(expression: &str) -> Result<ExpressionEvaluator<Complex<f64>>> {
    let expr = try_parse!(expression).map_err(|e| anyhow!(e))?;
    let param = try_parse!(PARAM_NAME).map_err(|e| anyhow!(e))?;
    let var_v = try_parse!("v").map_err(|e| anyhow!(e))?;
    let var_p = try_parse!("p").map_err(|e| anyhow!(e))?;

    let mut fn_map = FunctionMap::new();
    fn_map.add_constant(var_v, parse_complex_rational(V_VALUE)?);
    fn_map.add_constant(var_p, parse_complex_rational(P_VALUE)?);

    Atom::evaluator_multiple(&[expr], &fn_map, &[param], OptimizationSettings::default())
        .map(|eval| {
            eval.map_coeff(&|r: &float::Complex<Fraction<IntegerRing>>| {
                Complex::new(r.re.to_f64(), r.im.to_f64())
            })
        })
        .map_err(|e| anyhow!(e))
}

fn test_ifelse() -> Result<()> {
    let expression_source = load_expression(&PathBuf::from("expression.txt"));
    let input = [PARAM_VALUE];

    let symjit_eval = build_evaluator(&expression_source)?;
    // println!("{:?}", symjit_eval.export_instructions());
    // let app = CompiledComplexRunner::compile(&symjit_eval, config)?.seal()?;
    let config = Config::default();
    let mut app = InterpretedComplexRunner::compile(&symjit_eval, config)?;
    let mut out = vec![Complex::new(0.0, 0.0)];

    app.app.dump("test.bin", "bytecode");

    app.evaluate(&input, &mut out);
    println!("ifelse output: {:?}", &out);

    Ok(())
}

/* ************************************************ */

pub fn main() -> Result<()> {
    test_real()?;
    pass("real");

    test_complex()?;
    pass("complex");

    test_real_runner()?;
    pass("real runner");

    test_complex_runner()?;
    pass("complex runner");

    test_scattered_simd_real_runner()?;
    pass("Scattered simd real runner");

    test_scattered_simd_complex_runner()?;
    pass("Scattered simd complex runner");

    test_interpreted_real_runner()?;
    pass("interpreted real runner");

    test_interpreted_complex_runner()?;
    pass("interpreted complex runner");

    test_external()?;
    pass("external real runner");

    test_external_save()?;
    test_external_load()?;
    pass("external func real runner (save/load)");

    test_external_func_complex()?;
    pass("external func complex runner");

    // test_external_func_bytecode()?;
    // pass("external func bytecode runner");
    #[cfg(target_arch = "x86_64")]
    test_external_simd_func()?;
    pass("external func simd runner");

    #[cfg(target_arch = "x86_64")]
    test_external_simd_complex_func()?;
    pass("external func simd complex runner");

    test_string_real()?;
    pass("string real runner");

    test_string_complex()?;
    pass("string complex runner");

    test_threads_runner()?;
    pass("threads");

    test_threads_application()?;
    pass("threads");

    test_ifelse()?;
    pass("ifelse");

    Ok(())
}
