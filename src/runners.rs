use crate::{compile, compile_string};
use anyhow::Result;
use symbolica::evaluate::ExpressionEvaluator;
use symjit::Storage;
pub use symjit::{Applet, Application, Complex, Config, Defuns, Element};

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
    pub app: Application,
}

impl CompiledRealRunner {
    pub fn compile(ev: &ExpressionEvaluator<f64>, config: Config) -> Result<Self> {
        Self::compile_with_funcs(ev, config, Defuns::new(), 0)
    }

    pub fn compile_with_funcs(
        ev: &ExpressionEvaluator<f64>,
        mut config: Config,
        df: Defuns,
        num_params: usize,
    ) -> Result<Self> {
        config.set_complex(false);
        config.set_simd(true);
        let app = compile(&ev, config, df, num_params)?;
        Ok(Self { app })
    }

    pub fn compile_string(model: String, config: Config) -> Result<Self> {
        Self::compile_string_with_funcs(model, config, Defuns::new(), 0)
    }

    pub fn compile_string_with_funcs(
        model: String,
        mut config: Config,
        df: Defuns,
        num_params: usize,
    ) -> Result<Self> {
        config.set_complex(false);
        config.set_simd(true);
        let app = compile_string(model, config, df, num_params)?;
        Ok(Self { app })
    }

    pub fn evaluate<T>(&self, args: &[T], outs: &mut [T])
    where
        T: Element,
    {
        let n = args.len() / self.app.count_params;
        assert!(outs.len() / self.app.count_obs >= n);
        self.app.evaluate_matrix(args, outs, n);
    }

    pub fn save(&self, file: &str) -> Result<()> {
        let mut fs = std::fs::File::create(file)?;
        self.app.save(&mut fs)
    }

    pub fn load(file: &str) -> Result<Self> {
        let mut fs = std::fs::File::open(file)?;
        let app = Application::load(&mut fs)?;
        Ok(Self { app })
    }

    pub fn seal(self) -> Result<Applet> {
        self.app.seal()
    }
}

/************************ CompiledComplexRunner ***************************/

pub struct CompiledComplexRunner {
    pub app: Application,
}

impl CompiledComplexRunner {
    pub fn compile(ev: &ExpressionEvaluator<Complex<f64>>, config: Config) -> Result<Self> {
        Self::compile_with_funcs(ev, config, Defuns::new(), 0)
    }

    pub fn compile_with_funcs(
        ev: &ExpressionEvaluator<Complex<f64>>,
        mut config: Config,
        df: Defuns,
        num_params: usize,
    ) -> Result<Self> {
        config.set_complex(true);
        config.set_simd(true);
        let app = compile(&ev, config, df, num_params)?;
        Ok(CompiledComplexRunner { app })
    }

    pub fn compile_string(model: String, config: Config) -> Result<Self> {
        Self::compile_string_with_funcs(model, config, Defuns::new(), 0)
    }

    pub fn compile_string_with_funcs(
        model: String,
        mut config: Config,
        df: Defuns,
        num_params: usize,
    ) -> Result<Self> {
        config.set_complex(true);
        config.set_simd(true);
        let app = compile_string(model, config, df, num_params)?;
        Ok(CompiledComplexRunner { app })
    }

    pub fn evaluate<T>(&self, args: &[T], outs: &mut [T])
    where
        T: Element,
    {
        let n = (2 * args.len()) / self.app.count_params;
        assert!(2 * outs.len() / self.app.count_obs >= n);
        self.app.evaluate_matrix(args, outs, n);
    }

    pub fn save(&self, file: &str) -> Result<()> {
        let mut fs = std::fs::File::create(file)?;
        self.app.save(&mut fs)
    }

    pub fn load(file: &str) -> Result<Self> {
        let mut fs = std::fs::File::open(file)?;
        let app = Application::load(&mut fs)?;
        Ok(Self { app })
    }

    pub fn seal(self) -> Result<Applet> {
        self.app.seal()
    }
}

/********************* InterpretedRealRunner ************************/

pub struct InterpretedRealRunner {
    app: Application,
}

impl InterpretedRealRunner {
    pub fn compile(ev: &ExpressionEvaluator<f64>, config: Config) -> Result<Self> {
        Self::compile_with_funcs(ev, config, Defuns::new(), 0)
    }

    pub fn compile_with_funcs(
        ev: &ExpressionEvaluator<f64>,
        config: Config,
        df: Defuns,
        num_params: usize,
    ) -> Result<Self> {
        let mut c = Config::from_name("bytecode", config.opt)?;
        c.set_complex(false);
        c.set_simd(false);
        let app = compile(&ev, c, df, num_params)?;
        Ok(Self { app })
    }

    pub fn compile_string(model: String, config: Config) -> Result<Self> {
        Self::compile_string_with_funcs(model, config, Defuns::new(), 0)
    }

    pub fn compile_string_with_funcs(
        model: String,
        config: Config,
        df: Defuns,
        num_params: usize,
    ) -> Result<Self> {
        let mut c = Config::from_name("bytecode", config.opt)?;
        c.set_complex(false);
        c.set_simd(false);
        let app = compile_string(model, c, df, num_params)?;
        Ok(Self { app })
    }

    pub fn evaluate(&mut self, args: &[f64], outs: &mut [f64]) {
        let n = args.len() / self.app.count_params;
        assert!(outs.len() / self.app.count_obs >= n);
        self.app.interpret_matrix(args, outs, n);
    }

    pub fn save(&self, file: &str) -> Result<()> {
        let mut fs = std::fs::File::create(file)?;
        self.app.save(&mut fs)
    }

    pub fn load(file: &str) -> Result<Self> {
        let mut fs = std::fs::File::open(file)?;
        let app = Application::load(&mut fs)?;
        Ok(Self { app })
    }
}

/********************* InterpretedComplexRunner ************************/

pub struct InterpretedComplexRunner {
    app: Application,
}

impl InterpretedComplexRunner {
    pub fn compile(ev: &ExpressionEvaluator<Complex<f64>>, config: Config) -> Result<Self> {
        Self::compile_with_funcs(ev, config, Defuns::new(), 0)
    }

    pub fn compile_with_funcs(
        ev: &ExpressionEvaluator<Complex<f64>>,
        config: Config,
        df: Defuns,
        num_params: usize,
    ) -> Result<Self> {
        let mut c = Config::from_name("bytecode", config.opt)?;
        c.set_complex(true);
        c.set_simd(false);
        let app = compile(&ev, c, df, num_params)?;
        Ok(Self { app })
    }

    pub fn compile_string(model: String, config: Config) -> Result<Self> {
        Self::compile_string_with_funcs(model, config, Defuns::new(), 0)
    }

    pub fn compile_string_with_funcs(
        model: String,
        config: Config,
        df: Defuns,
        num_params: usize,
    ) -> Result<Self> {
        let mut c = Config::from_name("bytecode", config.opt)?;
        c.set_complex(true);
        c.set_simd(false);
        let app = compile_string(model, c, df, num_params)?;
        Ok(Self { app })
    }

    pub fn evaluate(&mut self, args: &[Complex<f64>], outs: &mut [Complex<f64>]) {
        let n = (2 * args.len()) / self.app.count_params;
        assert!((2 * outs.len()) / self.app.count_obs >= n);

        let args = flatten_vec(args);
        let outs = flatten_vec_mut(outs);

        self.app.interpret_matrix(args, outs, n);
    }

    pub fn save(&self, file: &str) -> Result<()> {
        let mut fs = std::fs::File::create(file)?;
        self.app.save(&mut fs)
    }

    pub fn load(file: &str) -> Result<Self> {
        let mut fs = std::fs::File::open(file)?;
        let app = Application::load(&mut fs)?;
        Ok(Self { app })
    }
}
