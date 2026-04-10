use anyhow::Result;
use std::time::Instant;

use symbolica::{
    domains::{float::Complex, rational::Rational},
    evaluate::ExpressionEvaluator,
};

use num_complex;
use pyo3;
use rand::prelude::*;
use symjit_bridge::{CompiledComplexRunner, Config};

fn main() -> Result<()> {
    let t0 = Instant::now();
    let b = std::fs::read("../huge/large_poly_evaluator.dat")?;
    let t1 = Instant::now();

    let eval: ExpressionEvaluator<Complex<Rational>> =
        bincode::decode_from_slice(&b, bincode::config::standard())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?
            .0;

    let eval = eval.map_coeff(&|x| num_complex::Complex::new(x.re.to_f64(), x.im.to_f64()));

    let t2 = Instant::now();

    let mut config = Config::default();
    config.set_complex(true);
    config.set_simd(false);
    config.set_mem_saver(true);
    config.set_permissive(true);

    let runner = CompiledComplexRunner::compile(&eval, config)?;
    let app = runner.seal()?;

    let t3 = Instant::now();

    let mut args: Vec<num_complex::Complex<f64>> = vec![num_complex::Complex::default(); 8];
    let mut outs: Vec<num_complex::Complex<f64>> = vec![num_complex::Complex::default(); 1];

    let mut rng = rand::rng();

    for i in 0..100 {
        println!("run {}: {:?}", i, &outs);

        for j in 0..8 {
            args[j] =
                num_complex::Complex::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5);
            app.evaluate(&args, &mut outs);
        }
    }

    let t4 = Instant::now();

    println!("loading:\t{:?}", t1.duration_since(t0));
    println!("decoding:\t{:?}", t2.duration_since(t1));
    println!("compiling:\t{:?}", t3.duration_since(t2));
    println!("running:\t{:?} (100 runs)", t4.duration_since(t3));
    println!("bytes: {}", app.compiled.map_or(0, |m| m.size));

    Ok(())
}
