//! Configuración del Thread Pool Global para Rayon y Polars
//!
//! Permite fijar el número exacto de hilos de trabajo paralelos (ej: 20 hilos)
//! para acelerar el procesamiento de datos, cálculo de indicadores e inferencia.

use std::sync::atomic::{AtomicUsize, Ordering};

pub struct ThreadingConfig;

static CONFIGURED_THREADS: AtomicUsize = AtomicUsize::new(20);

impl ThreadingConfig {
    pub const DEFAULT_THREADS: usize = 20;

    /// Inicializa el pool global de Rayon y variables de entorno para Polars y BLAS
    pub fn initialize_global_pool(threads: usize) {
        let n_threads = if threads == 0 { Self::DEFAULT_THREADS } else { threads };
        CONFIGURED_THREADS.store(n_threads, Ordering::SeqCst);

        // Variables de entorno para Polars, Rayon y librerías numéricas
        std::env::set_var("RAYON_NUM_THREADS", n_threads.to_string());
        std::env::set_var("POLARS_MAX_THREADS", n_threads.to_string());
        std::env::set_var("OMP_NUM_THREADS", n_threads.to_string());

        // Configuración del Pool Global de Rayon
        let result = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .thread_name(|idx| format!("gbt-worker-{}", idx))
            .build_global();

        match result {
            Ok(_) => {
                println!("🚀 Thread Pool paralelizado configurado con exito: {} hilos activos.", n_threads);
            }
            Err(e) => {
                // Si el pool ya fue inicializado previamente en la misma sesión, Rayon no permite recrearlo
                println!("ℹ️ Thread Pool global ya activo ({} hilos): {}", n_threads, e);
            }
        }
    }

    /// Devuelve el número de hilos configurados
    pub fn num_threads() -> usize {
        CONFIGURED_THREADS.load(Ordering::SeqCst)
    }
}
