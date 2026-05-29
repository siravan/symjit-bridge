use anyhow::{Result, anyhow};
use symjit::{CompilerType, Composer, Complex, Config, Slot, Translator};

// 37_500 succeeds on Apple Silicon in this setup; 40_000 fails.
const TERMS: usize = 40_000;

fn main() -> Result<()> {
    let mut config = Config::new(CompilerType::Native, 0).map_err(|e| anyhow!(e))?;
    config.set_opt_level(0);
    config.set_simd(true);

    let mut translator = Translator::new(config);
    translator.set_num_params(1);

    let zero = Slot::Const(
        translator
            .append_constant(Complex::new(0.0, 0.0))
            .map_err(|e| anyhow!(e))?,
    );
    let one = Slot::Const(
        translator
            .append_constant(Complex::new(1.0, 0.0))
            .map_err(|e| anyhow!(e))?,
    );
    let x = Slot::Param(0);

    translator
        .append_if_else(&x, 0)
        .map_err(|e| anyhow!(e))?;

    for i in 0..TERMS {
        translator
            .append_add(&Slot::Temp(i), &[x, one], 2)
            .map_err(|e| anyhow!(e))?;
    }

    translator.append_label(0).map_err(|e| anyhow!(e))?;
    translator
        .append_assign(&Slot::Out(0), &zero)
        .map_err(|e| anyhow!(e))?;

    let _code = translator
        .compile()
        .map_err(|e| anyhow!(e))?
        .seal()
        .map_err(|e| anyhow!(e))?;

    Ok(())
}
