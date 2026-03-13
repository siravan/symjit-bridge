use anyhow::Result;
use rand::prelude::*;
use symjit_bridge::{CompiledRealRunner, Config};

fn main() -> Result<()> {
    const NROWS: usize = 97;

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

    println!("passed!");

    Ok(())
}
