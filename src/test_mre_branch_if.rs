use anyhow::{anyhow, Result};
use symjit::{CompilerType, Complex, Composer, Config, Slot, Translator};

fn generate(nt: usize) -> Result<()> {
    let mut config = Config::new(CompilerType::Native, 0)?;
    config.set_opt_level(0);
    config.set_simd(true);
    config.set_complex(true);
    config.set_dicect(true);

    let mut translator = Translator::new(config);
    translator.set_num_params(1);

    let zero = Slot::Const(translator.append_constant(Complex::new(0.0, 0.0))?);
    let one = Slot::Const(translator.append_constant(Complex::new(1.0, 0.0))?);

    let x = Slot::Param(0);

    translator.append_if_else(&x, 0)?;
    translator.append_add(&Slot::Temp(0), &[one, zero], 2)?;

    for i in 1..nt {
        translator.append_add(&Slot::Temp(i), &[Slot::Temp(i - 1), x], 2)?;
    }

    translator.append_goto(1)?;
    translator.append_label(0)?;
    translator.append_assign(&Slot::Temp(nt), &one)?;
    translator.append_label(1)?;
    translator.append_join(&Slot::Out(0), &x, &Slot::Temp(nt - 1), &Slot::Temp(nt))?;

    let code = translator.compile()?.seal()?;

    let args = vec![Complex::new(1.0, 0.0)];
    let mut outs = vec![Complex::new(0.0, 0.0)];

    code.evaluate(&args, &mut outs);

    let expectation = Complex::new(nt as f64, 0.0);

    if outs[0] != expectation {
        Err(anyhow!("expects {}, obsered {}", expectation, outs[0]))
    } else {
        Ok(())
    }
}

fn main() -> Result<()> {
    for n in 1..19 {
        let nt = 1 << n;
        print!("testing TERMS = {}...", nt);
        generate(nt).map_err(|e| anyhow!(e))?;
        println!("pass!");
    }

    Ok(())
}
