use std::time::Duration;
use reqwest::Client;
use crate::types::{Kline, Timeframe};
use super::endpoint_pool::EndpointPool;
use super::raw_binance_kline::RawBinanceKline;

/// Lote estructurado con el resultado de la descarga y el conteo bruto de Binance
#[derive(Debug, Clone, PartialEq)]
pub struct KlineBatch {
    pub klines: Vec<Kline>,
    pub raw_count: usize,
}

#[derive(Clone)]
pub struct BinanceClient {
    client: Client,
    endpoints: EndpointPool,
}

impl BinanceClient {
    pub fn new() -> Result<Self, reqwest::Error> {
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .build()?;

        Ok(Self {
            client,
            endpoints: EndpointPool::default(),
        })
    }

    pub async fn download_klines(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start_time: Option<i64>,
        limit: u32,
    ) -> Result<KlineBatch, Box<dyn std::error::Error + Send + Sync>> {
        let mut last_err = None;
        let limit = limit.min(1000);

        // BUG-20: Rotación atómica de endpoints para evitar reintentar en endpoints bloqueados con HTTP 429
        for base_url in self.endpoints.rotated_endpoints() {
            let mut url = format!(
                "{}/api/v3/klines?symbol={}&interval={}&limit={}",
                base_url,
                symbol,
                timeframe.as_binance_str(),
                limit
            );

            if let Some(st) = start_time {
                url.push_str(&format!("&startTime={}", st));
            }

            match self.client.get(&url).send().await {
                Ok(res) => {
                    if res.status().is_success() {
                        match res.text().await {
                            Ok(body) => match serde_json::from_str::<Vec<RawBinanceKline>>(&body) {
                                Ok(raw_list) => {
                                    let raw_count = raw_list.len();
                                    // BUG-30: Filtro estricto anti Look-Ahead con margen de seguridad temporal (safety buffer de 1500 ms)
                                    // para prevenir contaminación por desincronización de reloj local (NTP Drift)
                                    let now_ms = chrono::Utc::now().timestamp_millis();
                                    let safe_cutoff = now_ms.saturating_sub(1500);
                                    let klines: Vec<Kline> = raw_list
                                        .into_iter()
                                        .filter(|raw| raw.close_time() <= safe_cutoff)
                                        .filter_map(|raw| raw.to_kline())
                                        .collect();
                                    return Ok(KlineBatch { klines, raw_count });
                                }
                                Err(e) => {
                                    last_err = Some(format!(
                                        "Error deserializando klines de {}: {} - Resp: {}",
                                        base_url, e, body
                                    ));
                                }
                            },
                            Err(e) => {
                                last_err = Some(format!("Error leyendo cuerpo HTTP de {}: {}", base_url, e));
                            }
                        }
                    } else {
                        last_err = Some(format!(
                            "Status HTTP de error en {}: {}",
                            base_url,
                            res.status()
                        ));
                    }
                }
                Err(e) => {
                    last_err = Some(format!("Fallo de conexión con {}: {}", base_url, e));
                }
            }

            tokio::time::sleep(Duration::from_millis(300)).await;
        }

        Err(format!(
            "Todos los endpoints de Binance fallaron para {} {}. Último error: {:?}",
            symbol,
            timeframe.as_binance_str(),
            last_err
        )
        .into())
    }
}
