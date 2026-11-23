use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MarginMode {
    /// Todo el capital de la cuenta respalda la posición. 
    /// Una liquidación consume todo el balance libre.
    Cross,
    /// Solo el margen comprometido en la operación específica está en riesgo.
    /// Si la pérdida alcanza el margen asignado, solo se pierde dicho margen.
    #[default]
    Isolated,
}

impl std::fmt::Display for MarginMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarginMode::Cross => write!(f, "Cross (Cruzado)"),
            MarginMode::Isolated => write!(f, "Isolated (Aislado)"),
        }
    }
}
