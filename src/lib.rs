use anyhow::Result;
use num_complex::Complex;
use symjit::{compiler, Translator};

pub use symjit::{Application, Config};

use symbolica::evaluate::{BuiltinSymbol, ExpressionEvaluator, Instruction, Slot};

fn slot(s: Slot) -> compiler::Slot {
    match s {
        Slot::Param(id) => compiler::Slot::Param(id),
        Slot::Out(id) => compiler::Slot::Out(id),
        Slot::Const(id) => compiler::Slot::Const(id),
        Slot::Temp(id) => compiler::Slot::Temp(id),
    }
}

fn slot_list(v: &[Slot]) -> Vec<compiler::Slot> {
    v.iter().map(|s| slot(*s)).collect::<Vec<compiler::Slot>>()
}

fn builtin_symbol(s: BuiltinSymbol) -> compiler::BuiltinSymbol {
    compiler::BuiltinSymbol(s.get_symbol().get_id())
}

fn translate(
    instructions: Vec<Instruction>,
    constants: Vec<Complex<f64>>,
    config: Config,
) -> Result<Translator> {
    let mut translator = Translator::new(config);

    for z in constants {
        translator.append_constant(z)?;
    }

    for q in instructions {
        match q {
            Instruction::Add(lhs, args, num_reals) => {
                translator.append_add(&slot(lhs), &slot_list(&args), num_reals)?
            }
            Instruction::Mul(lhs, args, num_reals) => {
                translator.append_mul(&slot(lhs), &slot_list(&args), num_reals)?
            }
            Instruction::Pow(lhs, arg, p, is_real) => {
                translator.append_pow(&slot(lhs), &slot(arg), p, is_real)?
            }
            Instruction::Powf(lhs, arg, p, is_real) => {
                translator.append_powf(&slot(lhs), &slot(arg), &slot(p), is_real)?
            }
            Instruction::Assign(lhs, rhs) => translator.append_assign(&slot(lhs), &slot(rhs))?,
            Instruction::Fun(lhs, fun, arg, is_real) => {
                translator.append_fun(&slot(lhs), &builtin_symbol(fun), &slot(arg), is_real)?
            }
            Instruction::Join(lhs, cond, true_val, false_val) => translator.append_join(
                &slot(lhs),
                &slot(cond),
                &slot(true_val),
                &slot(false_val),
            )?,
            Instruction::Label(id) => translator.append_label(id)?,
            Instruction::IfElse(cond, id) => translator.append_if_else(&slot(cond), id)?,
            Instruction::Goto(id) => translator.append_goto(id)?,
            Instruction::ExternalFun(lhs, op, args) => {
                translator.append_external_fun(&slot(lhs), &op, &slot_list(&args))?
            }
        }
    }

    Ok(translator)
}

pub trait Number {
    fn as_complex(&self) -> Complex<f64>;
}

impl Number for Complex<f64> {
    fn as_complex(&self) -> Complex<f64> {
        *self
    }
}

impl Number for f64 {
    fn as_complex(&self) -> Complex<f64> {
        Complex::new(*self, 0.0)
    }
}

pub fn compile<T: Clone + Number>(
    ev: &ExpressionEvaluator<T>,
    config: Config,
) -> Result<Application> {
    let (instructions, _, constants) = ev.export_instructions();
    let constants: Vec<Complex<f64>> = constants.iter().map(|x| x.as_complex()).collect();
    let mut translator = translate(instructions, constants, config).unwrap();
    translator.compile()
}
