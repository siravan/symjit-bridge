use anyhow::Result;

use symjit_bridge::{
    compile, CompiledComplexRunner, CompiledRealRunner, Complex, ComplexFloat, Config, Defuns,
    InterpretedComplexRunner, InterpretedRealRunner,
};

use symbolica::{
    atom::AtomCore,
    evaluate::{FunctionMap, OptimizationSettings},
    parse, symbol,
};

use rand::prelude::*;

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

    let mut app = compile(&ev, Config::default(), &Defuns::new(), 0)?;
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
    let mut app = compile(&ev, config, &Defuns::new(), 0)?;
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

fn test_scattered_simd_real_runner() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x + y^3")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let mut runner = CompiledRealRunner::compile(&ev, Config::default())?;
    let args: Vec<f64> = (0..8).map(|x| f64::from(x)).collect();
    let mut outs = [0.0; 4];
    runner.evaluate(&args, &mut outs);
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

    const NROWS: i32 = 97;

    let mut runner = CompiledComplexRunner::compile(&ev, Config::default())?;
    let args: Vec<Complex<f64>> = (0..NROWS * 2)
        .map(|x| Complex::new(f64::from(x), -1.0))
        .collect();
    let mut outs = [Complex::<f64>::default(); NROWS as usize];
    runner.evaluate(&args, &mut outs);

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

    let mut runner = CompiledComplexRunner::compile(&ev, Config::default())?;
    let args = [Complex::new(1.0, 2.0), Complex::new(2.0, -1.0)];
    let mut outs = [Complex::<f64>::default(); 1];
    runner.evaluate(&args, &mut outs);
    assert_eq!(outs[0], Complex::new(3.0, 1.0).sinh());
    Ok(())
}

extern "C" fn test(x: f64, y: f64) -> f64 {
    f64::sin(x + y)
}

fn fun(x: &[f64]) -> f64 {
    x.iter().sum()
}

fn test_external_func() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let mut f = FunctionMap::new();
    f.add_external_function(symbol!("test"), "test".to_string())
        .unwrap();

    let ev = parse!("test(x, y, -(x + y))")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let mut df = Defuns::new();
    df.add_sliced_func("test", |x: &[f64]| x.iter().sum());

    let mut config = Config::default();
    config.set_opt_level(2);

    //let config = Config::from_name("bytecode", Config::default().opt)?;
    let config = Config::default();
    let mut runner = CompiledRealRunner::compile_with_funcs(&ev, config, &df, 0)?;

    runner.app.dump("ext.bin", "scalar");

    const N: usize = 117;
    let args: Vec<f64> = (0..N * 2).map(|x| x as f64).collect();
    let mut outs: Vec<f64> = vec![0.0; N];
    runner.evaluate(&args, &mut outs);

    for i in 0..N {
        assert_eq!(outs[i], 0.0);
    }

    Ok(())
}

extern "C" fn cplx_test(x: f64, y: f64, z: &mut Complex<f64>) {
    *z = (Complex::new(x, y) + *z).sinh();
}

fn fun_complex(x: &[Complex<f64>]) -> Complex<f64> {
    let s: Complex<f64> = x.iter().sum();
    s + Complex::new(1.0, 0.0)
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

    let mut df = Defuns::new();
    // df.add_binary_complex("test", cplx_test);
    df.add_sliced_func("test", fun_complex);

    let mut runner = CompiledComplexRunner::compile_with_funcs(&ev, Config::default(), &df, 0)?;

    let mut rng = rand::rng();
    const N: usize = 223;

    let args: Vec<Complex<f64>> = (0..N * 2)
        .map(|_| Complex::new(rng.random(), rng.random()))
        .collect();
    let mut outs: Vec<Complex<f64>> = vec![Complex::default(); N];

    runner.evaluate(&args, &mut outs);
    assert_eq!(outs[0], Complex::new(1.0, 0.0));
    Ok(())
}

fn test_string_real() -> Result<()> {
    const NROWS: usize = 97;

    /*
    *  test_instructions.txt is generated as:
    *
       s = str(E("if(x, 3*sin(y)^2, 3*sin(z)^2) + if(x, 3*cos(y)^2, 3*cos(z)^2)")
           .evaluator({}, {}, [S("x"), S("y"), S("z")],
           conditionals=[S("if")]).get_instructions())
    */
    let model = std::fs::read_to_string("test_instructions.txt")?;
    let mut runner = CompiledRealRunner::compile_string(model, Config::default())?;

    let mut args: Vec<f64> = vec![0.0; NROWS * 3];
    let mut rng = rand::rng();

    for i in 0..NROWS {
        args[i * 3] = if rng.random::<f64>() < 0.5 { 0.0 } else { 1.0 };
        args[i * 3 + 1] = rng.random::<f64>();
        args[i * 3 + 2] = rng.random::<f64>();
    }

    let mut outs = [0.0; NROWS as usize];
    runner.evaluate(&args, &mut outs);

    for i in 0..NROWS {
        let delta = outs[i] - 3.0;
        assert!(delta.abs() < 1e-15);
    }

    Ok(())
}

fn test_string_complex() -> Result<()> {
    const NROWS: usize = 97;

    /*
    *  test_instructions.txt is generated as:
    *
       s = str(E("if(x, 3*sin(y)^2, 3*sin(z)^2) + if(x, 3*cos(y)^2, 3*cos(z)^2)")
           .evaluator({}, {}, [S("x"), S("y"), S("z")],
           conditionals=[S("if")]).get_instructions())
    */
    let model = std::fs::read_to_string("test_instructions.txt")?;
    let mut runner = CompiledComplexRunner::compile_string(model, Config::default())?;

    let mut args: Vec<Complex<f64>> = vec![Complex::default(); NROWS * 3];
    let mut rng = rand::rng();

    for i in 0..NROWS {
        args[i * 3] = if rng.random::<f64>() < 0.5 {
            Complex::default()
        } else {
            Complex::new(1.0, 0.0)
        };
        args[i * 3 + 1] = Complex::new(rng.random::<f64>(), rng.random::<f64>());
        args[i * 3 + 2] = Complex::new(rng.random::<f64>(), rng.random::<f64>());
    }

    let mut outs = [Complex::default(); NROWS as usize];
    runner.evaluate(&args, &mut outs);

    for i in 0..NROWS {
        let delta = outs[i] - 3.0;
        assert!(delta.abs() < 1e-14);
    }

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

    test_scattered_simd_real_runner()?;
    pass("Scattered simd real runner");

    test_scattered_simd_complex_runner()?;
    pass("Scattered simd complex runner");

    test_interpreted_real_runner()?;
    pass("interpreted real runner");

    test_interpreted_complex_runner()?;
    pass("interpreted complex runner");

    //test_external()?;
    //pass("external real runner");

    test_external_func()?;
    pass("external func real runner");

    test_external_func_complex()?;
    pass("external func complex runner");

    test_string_real()?;
    pass("string real runner");

    test_string_complex()?;
    pass("string complex runner");

    Ok(())
}
