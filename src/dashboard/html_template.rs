//! Módulo de compatibilidad y punto de entrada para generación de Dashboards HTML.
//! La implementación detallada se encuentra modularizada en `crate::dashboard::template`.

pub use super::template::{escape_html, HtmlTemplate};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_html() {
        assert_eq!(
            escape_html("<script>alert('xss');</script>"),
            "&lt;script&gt;alert(&#39;xss&#39;);&lt;/script&gt;"
        );
        assert_eq!(
            escape_html("Tom & Jerry \"quotes\""),
            "Tom &amp; Jerry &quot;quotes&quot;"
        );
        assert_eq!(
            escape_html("Exit: SL hit < -2.0%"),
            "Exit: SL hit &lt; -2.0%"
        );
    }
}
