/// BUG-43: Función de escape de caracteres HTML para prevenir XSS o caracteres malformados en dashboards
pub fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

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
