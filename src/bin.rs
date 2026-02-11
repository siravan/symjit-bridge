use anyhow::Result;
use num_complex::Complex;
use symjit_bridge::{compile, Config};

use symbolica::{
    atom::AtomCore,
    evaluate::{FunctionMap, OptimizationSettings},
    parse, symbol,
};

use wide::{f64x2, f64x4};

fn assert_nearly_eq(x: f64, y: f64) {
    assert!((x - y).abs() < 1e-10);
}

fn pass(what: &str) {
    println!("**** test {:?} passed. ****", what);
}

fn test_compile() -> Result<()> {
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

fn test_symbolica_scalar() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];

    let mut f = FunctionMap::new();
    f.add_conditional(symbol!("if")).unwrap();

    let tests = vec![
        ("x + y^2", &[2.0, 5.0]),
        ("x - 4.0", &[-4.0, 10.0]),
        ("sin(x) + y", &[1.0, 5.0]),
        ("x^10 / y^3", &[2.0, 10.0]),
        ("if(y, x*x + y, x + 3)", &[5.0, 0.0]),
        ("if(y, x*x, x + 3)", &[5.0, 2.0]),
        ("if(y, x*x + y, x + 3)", &[5.0, 2.0]),
        ("x^2 + y^2", &[3.0, 4.0]),
        ("x^3 + y^3", &[5.0, 6.0]),
        ("x^30 + y^30", &[2.0, 3.0]),
    ];

    let config = Config::default();

    for (input, args) in tests {
        let mut ev = parse!(input)
            .evaluator(&f, &params, OptimizationSettings::default())
            .unwrap()
            .map_coeff(&|x| x.re.to_f64());

        let mut app = compile(&ev, config)?;
        assert_nearly_eq(app.evaluate_single(args), ev.evaluate_single(args));
    }

    Ok(())
}

fn test_symbolica_complex() -> Result<()> {
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

fn test_symbolica_external() -> Result<()> {
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
    assert_nearly_eq(u, f64::sinh(-1.0));

    Ok(())
}

fn test_symbolica_simd_f64x4() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let ev = parse!("x + y^2")
        .evaluator(
            &FunctionMap::new(),
            &params,
            OptimizationSettings::default(),
        )
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let mut config = Config::default();
    config.set_simd(true);
    let mut app = compile(&ev, config)?;

    let v = vec![
        f64x4::new([1.0, 2.0, 3.0, 4.0]),
        f64x4::new([5.0, 2.0, 1.0, 2.0]),
    ];
    let u = app.evaluate_simd_single(&v);

    assert_eq!(u, f64x4::new([26.0, 6.0, 4.0, 8.0]));

    Ok(())
}

/*
fn test_symbolica_simd_f64x2() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let eval = parse!("x + y^2")
        .evaluator(
            &FunctionMap::new(),
            &params,
            OptimizationSettings::default(),
        )
        .unwrap();

    let json = serde_json::to_string(&eval.export_instructions())?;
    let mut app = translate(&json, false, true)?;

    let v = vec![f64x2::new([1.0, 2.0]), f64x2::new([5.0, 2.0])];
    let u = app.evaluate_simd_single(&v);

    assert_eq!(u, f64x2::new([26.0, 6.0]));

    Ok(())
}
*/

fn test_symbolica_complex_simd_f64x4() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let ev = parse!("x + y^2")
        .evaluator(
            &FunctionMap::new(),
            &params,
            OptimizationSettings::default(),
        )
        .unwrap()
        .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

    let mut config = Config::default();
    config.set_complex(true);
    config.set_simd(true);
    let mut app = compile(&ev, config)?;

    let v = vec![
        Complex::new(
            f64x4::new([1.0, 2.0, 3.0, 4.0]),
            f64x4::new([3.0, 5.0, 2.0, -4.0]),
        ),
        Complex::new(
            f64x4::new([5.0, 2.0, 1.0, 2.0]),
            f64x4::new([-2.0, 1.0, 3.0, 7.0]),
        ),
    ];
    let u = app.evaluate_simd_single(&v);

    let res = Complex::new(
        f64x4::new([22.0, 5.0, -5.0, -41.0]),
        f64x4::new([-17.0, 9.0, 8.0, 24.0]),
    );

    assert_eq!(u, res);

    Ok(())
}

/*
fn test_symbolica_complex_simd_f64x2() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let eval = parse!("x + y^2")
        .evaluator(
            &FunctionMap::new(),
            &params,
            OptimizationSettings::default(),
        )
        .unwrap();

    let json = serde_json::to_string(&eval.export_instructions())?;
    let mut app = translate(&json, true, true)?;

    let v = vec![
        Complex::new(f64x2::new([1.0, 2.0]), f64x2::new([3.0, 5.0])),
        Complex::new(f64x2::new([5.0, 2.0]), f64x2::new([-2.0, 1.0])),
    ];
    let u = app.evaluate_simd_single(&v);

    let res = Complex::new(f64x2::new([22.0, 5.0]), f64x2::new([-17.0, 9.0]));

    assert_eq!(u, res);

    Ok(())
}
*/

fn test_symbolica_complex_matrix() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("cos(x^10 + y^10)")
        .evaluator(&f, &params, OptimizationSettings::default())
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

    assert_nearly_eq(outs[19].re, 1.0289805626427462);
    assert_nearly_eq(outs[19].im, -1.1072191382355374);

    Ok(())
}

fn test_symbolica_simd_matrix_f64x4() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let ev = parse!("x^5 - 4*x*y")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap()
        .map_coeff(&|x| x.re.to_f64());

    let mut config = Config::default();
    config.set_simd(true);
    let mut app = compile(&ev, config)?;

    const N: usize = 100;
    let mut input: Vec<f64x4> = vec![f64x4::default(); 2 * N];
    for i in 0..N {
        let x = i as f64;
        let y = 2.0 * i as f64;
        input[2 * i] = f64x4::new([x, x + 1.0, x + 2.0, x + 3.0]);
        input[2 * i + 1] = f64x4::new([x, x - 1.0, x - 2.0, x - 3.0]);
    }

    let mut outs: Vec<f64x4> = vec![f64x4::default(); N];
    app.evaluate_simd_matrix(&input, &mut outs, N);

    // note: 2474655 = 19^2 * (19^3 - 4) and so forth
    assert_eq!(
        outs[19],
        f64x4::new([2474655.0, 3198560.0, 4082673.0, 5152224.0])
    );

    Ok(())
}

/*
fn test_symbolica_simd_matrix_f64x2() -> Result<()> {
    let params = vec![parse!("x"), parse!("y")];
    let f = FunctionMap::new();
    let eval = parse!("x^5 - 4*x*y")
        .evaluator(&f, &params, OptimizationSettings::default())
        .unwrap();

    let json = serde_json::to_string(&eval.export_instructions())?;
    let mut app = translate(&json, false, true)?;

    const N: usize = 100;
    let mut input: Vec<f64x2> = vec![f64x2::default(); 2 * N];
    for i in 0..N {
        let x = i as f64;
        let y = 2.0 * i as f64;
        input[2 * i] = f64x2::new([x, x + 3.0]);
        input[2 * i + 1] = f64x2::new([x, x - 3.0]);
    }

    let mut outs: Vec<f64x2> = vec![f64x2::default(); N];
    app.evaluate_simd_matrix(&input, &mut outs, N);

    assert_eq!(outs[19], f64x2::new([2474655.0, 5152224.0]));

    Ok(())
}
*/

pub fn main() -> Result<()> {
    test_compile()?;
    pass("compile");

    test_symbolica_scalar()?;
    pass("scalar");

    test_symbolica_complex()?;
    pass("complex");

    test_symbolica_external()?;
    pass("external");

    test_symbolica_simd_f64x4()?;
    pass("simd");

    test_symbolica_complex_simd_f64x4()?;
    pass("complex simd");

    test_symbolica_complex_matrix()?;
    pass("complex matrix");

    test_symbolica_simd_matrix_f64x4()?;
    pass("simd matrix");

    Ok(())
}
