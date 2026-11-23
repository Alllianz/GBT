pub struct MenuPrinter;

impl MenuPrinter {
    pub fn print_main_banner() {
        println!();
        println!("===============================================================");
        println!("       🚀 GBT (Gradient Boosted Trader) - TERMINAL CUANTITATIVA ");
        println!("===============================================================");
        println!("  [1] DB                 - Gestión y Sincronización de Velas SQLite");
        println!("  [2] Forust-ML          - Gradient Boosted Decision Tree (Causal)");
        println!("  [3] TreeBoost          - Universal Boosting (LinearThenTree / AutoML)");
        println!("  [4] Native GBDT        - Gradient Boosting Propio Puro en Rust");
        println!("  [5] Random Forest      - Ensemble Bagging con Feature Subsampling");
        println!("  [6] Decision Tree CART - Árbol de Decisión Causal Determinista");
        println!("  [7] Baseline           - Estrategia Heurística de Referencia (EMA)");
        println!("  [8] Config             - Margen, Apalancamiento, Sizing, Métricas Fitness");
        println!("  [0] Salir");
        println!("===============================================================");
        print!("Selecciona una opción [0-8]: ");
    }

    pub fn print_db_menu() {
        println!();
        println!("---------------------------------------------------------------");
        println!("                 📁 MENÚ [1]: GESTIÓN DE VELAS DB             ");
        println!("---------------------------------------------------------------");
        println!("  [1] Ver inventario de velas guardadas por timeframe");
        println!("  [2] Descargar/Sincronizar velas para un timeframe específico");
        println!("  [3] Sincronización masiva de todos los timeframes");
        println!("  [0] Regresar al menú principal");
        println!("---------------------------------------------------------------");
        print!("Opción [0-3]: ");
    }

    pub fn print_forust_menu() {
        println!();
        println!("---------------------------------------------------------------");
        println!("             🌳 MENÚ [2]: FORUST-ML GBDT ENGINE               ");
        println!("---------------------------------------------------------------");
        println!("  [1] Backtest Causal Walk-Forward (IS -> OOS con Purga)");
        println!("  [2] Backtest Completo Purged (IS -> Purga/Cuarentena -> OOS Consolidado)");
        println!("  [3] Ver / Ajustar Hiperparámetros del modelo");
        println!("  [0] Regresar al menú principal");
        println!("---------------------------------------------------------------");
        print!("Opción [0-3]: ");
    }

    pub fn print_treeboost_menu() {
        println!();
        println!("---------------------------------------------------------------");
        println!("             ⚡ MENÚ [3]: TREEBOOST AUTO-ENGINE                ");
        println!("---------------------------------------------------------------");
        println!("  [1] Backtest Causal Walk-Forward (LinearThenTree IS -> OOS con Purga)");
        println!("  [2] Backtest Completo Purged (IS -> Purga/Cuarentena -> OOS Consolidado)");
        println!("  [3] Ver / Ajustar Hiperparámetros del modelo");
        println!("  [0] Regresar al menú principal");
        println!("---------------------------------------------------------------");
        print!("Opción [0-3]: ");
    }
}
