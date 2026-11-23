use rusqlite::Connection;
use crate::backtest::BacktestConfig;
use crate::models::ModelConfig;
use super::baseline_handler::BaselineHandler;
use super::cart_handler::CartHandler;
use super::config_handler::ConfigHandler;
use super::db_handler::DbHandler;
use super::forust_handler::ForustHandler;
use super::input_reader::InputReader;
use super::menu_printer::MenuPrinter;
use super::native_gbdt_handler::NativeGbdtHandler;
use super::random_forest_handler::RandomForestHandler;
use super::treeboost_handler::TreeBoostHandler;

pub struct InteractiveApp {
    model_config: ModelConfig,
    backtest_config: BacktestConfig,
}

impl Default for InteractiveApp {
    fn default() -> Self {
        Self::new()
    }
}

impl InteractiveApp {
    pub fn new() -> Self {
        Self {
            model_config: ModelConfig::institutional(),
            backtest_config: BacktestConfig::default(),
        }
    }

    pub async fn run(&mut self, conn: &mut Connection) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        loop {
            MenuPrinter::print_main_banner();
            let option = InputReader::read_line();

            match option.as_str() {
                "1" => {
                    // BUG-18: Capturar errores operacionales sin salir de la aplicación interactiva
                    if let Err(e) = DbHandler::handle(conn).await {
                        eprintln!("\n❌ Error durante la operación: {}\n", e);
                    }
                }
                "2" => {
                    if let Err(e) = ForustHandler::handle(conn, &mut self.model_config, &self.backtest_config) {
                        eprintln!("\n❌ Error durante la operación: {}\n", e);
                    }
                }
                "3" => {
                    if let Err(e) = TreeBoostHandler::handle(conn, &mut self.model_config, &self.backtest_config) {
                        eprintln!("\n❌ Error durante la operación: {}\n", e);
                    }
                }
                "4" => {
                    if let Err(e) = NativeGbdtHandler::handle(conn, &self.model_config, &self.backtest_config) {
                        eprintln!("\n❌ Error durante la operación: {}\n", e);
                    }
                }
                "5" => {
                    if let Err(e) = RandomForestHandler::handle(conn, &self.model_config, &self.backtest_config) {
                        eprintln!("\n❌ Error durante la operación: {}\n", e);
                    }
                }
                "6" => {
                    if let Err(e) = CartHandler::handle(conn, &self.model_config, &self.backtest_config) {
                        eprintln!("\n❌ Error durante la operación: {}\n", e);
                    }
                }
                "7" => {
                    if let Err(e) = BaselineHandler::handle(conn, &self.backtest_config) {
                        eprintln!("\n❌ Error durante la operación: {}\n", e);
                    }
                }
                "8" => {
                    ConfigHandler::handle(&mut self.backtest_config);
                }
                "0" => {
                    println!("\n👋 Finalizando sesión de GBT. ¡Hasta luego!");
                    break;
                }
                _ => {
                    println!("Opción no válida. Por favor, selecciona una opción entre 0 y 8.");
                }
            }
        }
        Ok(())
    }
}
