use anyhow::Result;
use num_complex::Complex;
use symjit::{Storage, Translator};
use wide::{f64x2, f64x4};

pub use symjit::{Application, Config};

use symbolica::evaluate::{BuiltinSymbol, ExpressionEvaluator, Instruction, Slot};

use crate::compile;

fn flatten_vec<T>(v: &[T]) -> &[f64] {
    let n = v.len();
    let p: *const f64 = unsafe { std::mem::transmute(v.as_ptr()) };
    let q: &[f64] = unsafe {
        std::slice::from_raw_parts(p, n * std::mem::size_of::<T>() / std::mem::size_of::<f64>())
    };
    q
}

fn flatten_vec_mut<T>(v: &mut [T]) -> &mut [f64] {
    let n = v.len();
    let p: *mut f64 = unsafe { std::mem::transmute(v.as_mut_ptr()) };
    let q: &mut [f64] = unsafe {
        std::slice::from_raw_parts_mut(p, n * std::mem::size_of::<T>() / std::mem::size_of::<f64>())
    };
    q
}

/********************* CompiledRealRunner ************************/

pub struct CompiledRealRunner {
    config: Config,
    app: Application,
}

impl CompiledRealRunner {
    pub fn compile(ev: &ExpressionEvaluator<f64>, mut config: Config) -> Result<Self> {
        config.set_complex(false);
        config.set_simd(false);
        let app = compile(&ev, config)?;
        Ok(Self { config, app })
    }

    pub fn evaluate(&mut self, args: &[f64], outs: &mut [f64]) {
        let n = args.len() / self.app.count_params;
        assert!(outs.len() / self.app.count_obs >= n);

        if self.config.use_threads() {
            self.app.evaluate_matrix_without_threads(args, outs, n);
        } else {
            self.app.evaluate_matrix_with_threads(args, outs, n);
        }
    }

    pub fn save(&self, file: &str) -> Result<()> {
        let mut fs = std::fs::File::create(file)?;
        self.app.save(&mut fs)
    }

    pub fn load(file: &str) -> Result<Self> {
        let mut fs = std::fs::File::open(file)?;
        let app = Application::load(&mut fs)?;
        let config = *app.prog.config();
        Ok(Self { config, app })
    }
}

/************************ CompiledComplexRunner ***************************/

pub struct CompiledComplexRunner {
    config: Config,
    app: Application,
}

impl CompiledComplexRunner {
    pub fn compile(ev: &ExpressionEvaluator<Complex<f64>>, mut config: Config) -> Result<Self> {
        config.set_complex(true);
        config.set_simd(false);
        let app = compile(&ev, config)?;
        Ok(CompiledComplexRunner { config, app })
    }

    pub fn evaluate(&mut self, args: &[Complex<f64>], outs: &mut [Complex<f64>]) {
        let n = (2 * args.len()) / self.app.count_params;
        assert!(2 * outs.len() / self.app.count_obs >= n);

        let args = flatten_vec(args);
        let outs = flatten_vec_mut(outs);

        if self.config.use_threads() {
            self.app.evaluate_matrix_without_threads(args, outs, n);
        } else {
            self.app.evaluate_matrix_with_threads(args, outs, n);
        }
    }

    pub fn save(&self, file: &str) -> Result<()> {
        let mut fs = std::fs::File::create(file)?;
        self.app.save(&mut fs)
    }

    pub fn load(file: &str) -> Result<Self> {
        let mut fs = std::fs::File::open(file)?;
        let app = Application::load(&mut fs)?;
        let config = *app.prog.config();
        Ok(Self { config, app })
    }
}

/**************************** CompiledSimdF64x4Runner ****************************/

pub struct CompiledSimdRealRunner {
    config: Config,
    app: Application,
}

impl CompiledSimdRealRunner {
    pub fn compile(ev: &ExpressionEvaluator<f64>, mut config: Config) -> Result<Self> {
        config.set_complex(false);
        config.set_simd(true);
        let app = compile(&ev, config)?;
        Ok(Self { config, app })
    }

    pub fn evaluate<T>(&mut self, args: &[T], outs: &mut [T]) {
        let n = args.len() / self.app.count_params;
        assert!(outs.len() / self.app.count_obs >= n);

        let args = flatten_vec(args);
        let outs = flatten_vec_mut(outs);

        if self.config.use_threads() {
            self.app
                .evaluate_matrix_without_threads_simd(args, outs, n, false);
        } else {
            self.app
                .evaluate_matrix_with_threads_simd(args, outs, n, false);
        }
    }

    pub fn save(&self, file: &str) -> Result<()> {
        let mut fs = std::fs::File::create(file)?;
        self.app.save(&mut fs)
    }

    pub fn load(file: &str) -> Result<Self> {
        let mut fs = std::fs::File::open(file)?;
        let app = Application::load(&mut fs)?;
        let config = *app.prog.config();
        Ok(Self { config, app })
    }
}

/**************************** CompiledSimdF64x4ComplexRunner ****************************/

pub struct CompiledSimdComplexRunner {
    config: Config,
    app: Application,
}

impl CompiledSimdComplexRunner {
    pub fn compile(ev: &ExpressionEvaluator<Complex<f64>>, mut config: Config) -> Result<Self> {
        config.set_complex(true);
        config.set_simd(true);
        let app = compile(&ev, config)?;
        Ok(Self { config, app })
    }

    pub fn evaluate<T>(&mut self, args: &[Complex<T>], outs: &mut [Complex<T>]) {
        let n = (2 * args.len()) / self.app.count_params;
        assert!(2 * outs.len() / self.app.count_obs >= n);

        let args = flatten_vec(args);
        let outs = flatten_vec_mut(outs);

        if self.config.use_threads() {
            self.app
                .evaluate_matrix_without_threads_simd(args, outs, n, false);
        } else {
            self.app
                .evaluate_matrix_with_threads_simd(args, outs, n, false);
        }
    }

    pub fn save(&self, file: &str) -> Result<()> {
        let mut fs = std::fs::File::create(file)?;
        self.app.save(&mut fs)
    }

    pub fn load(file: &str) -> Result<Self> {
        let mut fs = std::fs::File::open(file)?;
        let app = Application::load(&mut fs)?;
        let config = *app.prog.config();
        Ok(Self { config, app })
    }
}
