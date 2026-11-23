pub mod charts;
pub mod comparison_table;
pub mod escape;
pub mod layout;
pub mod metrics_cards;
pub mod styles;
pub mod trade_table;

pub use escape::escape_html;
use crate::backtest::{FullBacktestResult, WalkForwardResult};
use crate::types::FitnessMetric;

pub struct HtmlTemplate;

impl HtmlTemplate {
    pub fn render(wf: &WalkForwardResult) -> String {
        layout::render_full_dashboard(wf)
    }

    pub fn render_purged_full(
        model_name: &str,
        timeframe: &str,
        full: &FullBacktestResult,
        fitness_metric: FitnessMetric,
    ) -> String {
        layout::render_purged_full_dashboard(model_name, timeframe, full, fitness_metric)
    }
}
