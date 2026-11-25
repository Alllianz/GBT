pub mod technical_indicators;
pub mod feature_matrix;
pub mod h4_feature_matrix;
pub mod triple_barrier_labeler;
pub mod vol_scaled_labeler;
pub mod train_test_split;
pub mod embargo;
pub mod purged_kfold;
pub mod volatility;
pub mod selection;

pub use technical_indicators::TechnicalIndicators;
pub use feature_matrix::FeatureMatrix;
pub use h4_feature_matrix::H4FeatureMatrix;
pub use triple_barrier_labeler::{TripleBarrierLabeler, TargetLabel};
pub use vol_scaled_labeler::VolScaledLabeler;
pub use train_test_split::{TrainTestSplit, SplitDataset};
pub use embargo::TemporalEmbargo;
pub use purged_kfold::{PurgedKFold, PurgedFold};
pub use volatility::{ParkinsonVolatility, GarmanKlassVolatility, YangZhangVolatility};
pub use selection::{
    FeatureSelectionConfig, Xorshift64, RankCorrelationCalculator,
    CollinearityPruner, PurgedMdaEvaluator, MdaScore, CausalFeatureSelector,
};
