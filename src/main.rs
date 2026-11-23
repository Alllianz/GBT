use std::env;
use gbt::cli::InteractiveApp;
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
            "help" => {
                println!("Uso de GBT:");
                println!("  cargo run                -> Inicia la Terminal Cuantitativa Interactiva");
                println!("  cargo run -- interactive -> Inicia la Terminal Cuantitativa Interactiva");
            }
            _ => {
                println!("Comando desconocido: {}. Ejecuta 'cargo run' para la terminal interactiva.", args[1]);
            }
        }
    }

    Ok(())
}
