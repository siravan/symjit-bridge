use anyhow::Result;
use num_complex::{Complex, ComplexFloat};
use rand::{self, Rng};
use symjit_bridge::{CompiledComplexRunner, Config};

fn main() -> Result<()> {
    const NROWS: usize = 97;

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

    println!("passed!");

    Ok(())
}
