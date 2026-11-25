use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TradeRegime {
    #[default]
    PrimaryTrend,
    MeanReversionHedge,
}
