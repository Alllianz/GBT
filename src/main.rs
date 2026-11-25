use std::env;
use gbt::cli::{GbtBenchmarkRunner, GbtEvolveRunner, InteractiveApp};
use gbt::storage::{DbConfig, DbConnection};
use gbt::threading::ThreadingConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Inicialización del pool de 20 hilos para Rayon y Polars
    ThreadingConfig::initialize_global_pool(20);

    let args: Vec<String> = env::args().collect();
    let db_config = DbConfig::default();
    let mut conn = DbConnection::open(&db_config)?;

    if args.len() <= 1 {
        // Modo terminal interactiva por defecto
        let mut app = InteractiveApp::new();
        app.run(&mut conn).await?;
    } else {
        match args[1].as_str() {
            "interactive" => {
                let mut app = InteractiveApp::new();
                app.run(&mut conn).await?;
            }
            "bench-gbt" => {
                let tf_str = args
                    .get(2)
                    .ok_or("Uso: bench-gbt <1D|4H|1H> [--hedge] [--save-dashboard|--no-dashboard]")?;
                let mut hedge = false;
                let mut save_dashboard = true;
                for flag in &args[3..] {
                    match flag.as_str() {
                        "--hedge" => hedge = true,
                        "--save-dashboard" => save_dashboard = true,
                        "--no-dashboard" => save_dashboard = false,
                        _ => return Err(format!("Flag desconocido: {}", flag).into()),
                    }
                }
                GbtBenchmarkRunner::run(&conn, tf_str, hedge, save_dashboard)
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
            }
            "evolve-gbt" => {
                let tf_str = args.get(2).ok_or("Uso: evolve-gbt <4H|1H>")?;
                GbtEvolveRunner::run(&conn, tf_str)
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
            }
            "help" => {
                println!("Uso de GBT:");
                println!("  cargo run                -> Inicia la Terminal Cuantitativa Interactiva");
                println!("  cargo run -- interactive -> Inicia la Terminal Cuantitativa Interactiva");
                println!("  cargo run -- bench-gbt <1D|4H|1H> [--hedge] [--save-dashboard|--no-dashboard]");
                println!("  cargo run -- evolve-gbt <4H|1H>");
            }
            _ => {
                println!("Comando desconocido: {}. Ejecuta 'cargo run' para la terminal interactiva.", args[1]);
            }
        }
    }

    Ok(())
}
