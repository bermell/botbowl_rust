//! A forty-line blocking HTTP/1.1 client for `botbowl-hub job` to talk to
//! the daemon on `http://host:port`. No TLS (plan 040 phase 5 adds it, and
//! will replace this), no redirects, no chunked responses — axum sends
//! `content-length` for every JSON reply we read.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub fn request(method: &str, url: &str, token: &str, body: Option<&str>) -> io::Result<(u16, String)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("expected http://..., got {url}")))?;
    let (host, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let mut stream = TcpStream::connect(host)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let body = body.unwrap_or("");
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nhost: {host}\r\nauthorization: Bearer {token}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )?;
    let mut rd = BufReader::new(stream);
    let mut line = String::new();
    rd.read_line(&mut line)?;
    let status: u16 = line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("bad status line {line:?}")))?;
    let mut len: Option<usize> = None;
    loop {
        line.clear();
        rd.read_line(&mut line)?;
        let l = line.trim_end();
        if l.is_empty() {
            break;
        }
        if let Some(v) = l
            .split_once(':')
            .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        {
            len = v.1.trim().parse().ok();
        }
    }
    let mut out = Vec::new();
    match len {
        Some(n) => {
            out.resize(n, 0);
            rd.read_exact(&mut out)?;
        }
        None => {
            rd.read_to_end(&mut out)?;
        }
    }
    Ok((status, String::from_utf8_lossy(&out).into_owned()))
}
