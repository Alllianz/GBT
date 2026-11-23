use std::io::{self, Write};
use std::str::FromStr;

pub struct InputReader;

impl InputReader {
    pub fn read_line() -> String {
        io::stdout().flush().unwrap_or(());
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap_or(0);
        input.trim().to_string()
    }

    pub fn read_with_default<T: FromStr + ToString>(prompt: &str, default_val: T) -> T {
        print!("{} [por defecto: {}]: ", prompt, default_val.to_string());
        io::stdout().flush().unwrap_or(());

        let line = Self::read_line();
        if line.is_empty() {
            return default_val;
        }

        match line.parse::<T>() {
            Ok(val) => val,
            Err(_) => {
                println!("Entrada no válida, usando valor por defecto: {}", default_val.to_string());
                default_val
            }
        }
    }
}
