use anyhow::{Result, anyhow};
use symbolica::{
    atom::{Atom, AtomCore, FunctionBuilder},
    evaluate::{FunctionMap, OptimizationSettings},
    symbol,
};

// 8_125 succeeds on Apple Silicon in this setup; 8_500 fails.
const TERMS: usize = 8_500;

fn main() -> Result<()> {
    let x = Atom::var(symbol!("x"));
    let y = Atom::var(symbol!("y"));

    let terms = (1..=TERMS)
        .map(|i| x.pow(i as i64))
        .collect::<Vec<_>>();
    let large_branch = Atom::add_many(&terms);

    let expr = FunctionBuilder::new(symbol!("if"))
        .add_arg(&y)
        .add_arg(&large_branch)
        .add_arg(0)
        .finish();

    let mut function_map = FunctionMap::new();
    function_map
        .add_conditional(symbol!("if"))
        .map_err(|e| anyhow!(e))?;

    let settings = OptimizationSettings {
        horner_iterations: 0,
        direct_translation: true,
        ..OptimizationSettings::default()
    };

    let evaluator = expr
        .evaluator(&function_map, &[x, y], settings)
        .map_err(|e| anyhow!(e))?;

    let _jit = evaluator
        .jit_compile::<wide::f64x4>()
        .map_err(|e| anyhow!(e))?;

    Ok(())
}
