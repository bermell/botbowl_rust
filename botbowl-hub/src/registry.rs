//! `/registry/`: the project registry (`registry/*.md` — corpora, nets, experiments, matches),
//! browsable from the hub.
//!
//! The markdown files are the source; they are read from disk on every request, so an edit shows
//! on the next reload without restarting the hub. GitHub-flavoured tables render as tables
//! (`pulldown-cmark`), every heading gets an id (so `#exp069`-style links and the contents list
//! work), and wide tables scroll sideways instead of widening a phone's page. The pages are served
//! at `/registry/<FILE>.md`, so the files' own relative links (`[NETS.md](NETS.md)`) resolve to
//! the neighbouring page; `/registry/raw/<FILE>.md` is the file itself.
//!
//! The files are the repo's own, trusted content, so inline HTML in them (`<a id=…>`) is passed
//! through. Only names of `.md` files directly in the registry directory are ever opened.

use std::path::Path;
use std::time::SystemTime;

use pulldown_cmark::{CowStr, Event, HeadingLevel, Options, Parser, Tag, TagEnd};

use crate::page::{esc, nav, STYLE};

/// The registry's files in reading order; anything else in the directory follows by name.
const ORDER: [&str; 4] = ["DATA.md", "NETS.md", "EXPERIMENTS.md", "MATCHES.md"];

/// Extra style for rendered markdown: tables, headings, the filter box.
const DOC_STYLE: &str = "main{max-width:1400px}h1{font-size:1.3em;margin:.2em 0 .5em}\
     h2{font-size:1.15em;margin:1.4em 0 .4em;border-bottom:1px solid rgba(128,128,128,.35)}\
     h3{font-size:1.02em;margin:1.2em 0 .3em}\
     .tw{overflow-x:auto;margin:.5em 0 1em}\
     table{border-collapse:collapse;font-size:12px;line-height:1.35}\
     th,td{border:1px solid rgba(128,128,128,.35);padding:3px 6px;vertical-align:top;text-align:left}\
     th{background:rgba(128,128,128,.12);position:sticky;top:0}\
     tbody tr:nth-child(even){background:rgba(128,128,128,.05)}\
     details{margin:.4em 0 1em}summary{cursor:pointer;opacity:.85}\
     #q{font:inherit;padding:3px 6px;width:min(26em,100%);box-sizing:border-box;\
     background:transparent;color:inherit;border:1px solid rgba(128,128,128,.5);border-radius:4px}\
     .meta{opacity:.7;font-size:12px}li{margin:.15em 0}\
     @media (prefers-color-scheme:dark){th{background:rgba(255,255,255,.08)}}";

/// Hides table rows and list items that do not contain the filter text.
const FILTER_JS: &str = "<script>(function(){var q=document.getElementById('q');if(!q)return;\
     function f(){var t=q.value.toLowerCase(),n=0;\
     document.querySelectorAll('main tbody tr,main li').forEach(function(e){\
     var hit=!t||e.textContent.toLowerCase().indexOf(t)>=0;e.style.display=hit?'':'none';if(hit&&t)n++});\
     document.getElementById('qn').textContent=t?n+' match'+(n==1?'':'es'):''}\
     q.addEventListener('input',f);if(location.hash.indexOf('#q=')==0){q.value=decodeURIComponent(location.hash.slice(3));f()}})()</script>";

/// One registry file, for the index.
#[derive(Debug, Clone)]
pub struct Doc {
    /// `DATA.md`.
    pub file: String,
    /// Its first `# ` heading, else the file name.
    pub title: String,
    /// Its first paragraph after the title, as markdown.
    pub summary: String,
    pub modified: Option<SystemTime>,
}

/// A file name the registry will open: a `.md` file directly in the directory.
pub fn valid_name(name: &str) -> bool {
    name.len() > 3
        && name.ends_with(".md")
        && !name.starts_with('.')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
}

/// The registry's files, in reading order.
pub fn list(dir: &Path) -> Vec<Doc> {
    let mut docs: Vec<Doc> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let file = e.file_name().into_string().ok()?;
            if !valid_name(&file) || !e.file_type().ok()?.is_file() {
                return None;
            }
            let text = std::fs::read_to_string(e.path()).ok()?;
            let (title, summary) = title_and_summary(&text);
            Some(Doc {
                title: title.unwrap_or_else(|| file.clone()),
                summary,
                modified: e.metadata().ok()?.modified().ok(),
                file,
            })
        })
        .collect();
    docs.sort_by_key(|d| {
        (
            ORDER.iter().position(|o| *o == d.file).unwrap_or(ORDER.len()),
            d.file.clone(),
        )
    });
    docs
}

/// The first `# ` heading and the paragraph after it.
fn title_and_summary(md: &str) -> (Option<String>, String) {
    let mut lines = md.lines();
    let mut title = None;
    for l in lines.by_ref() {
        if let Some(t) = l.strip_prefix("# ") {
            title = Some(t.trim().to_string());
            break;
        }
        if !l.trim().is_empty() {
            // No title: the summary starts here.
            let rest: Vec<&str> = std::iter::once(l)
                .chain(md.lines().skip_while(|x| *x != l).skip(1))
                .take_while(|x| !x.trim().is_empty())
                .collect();
            return (None, rest.join("\n"));
        }
    }
    let para: Vec<&str> = lines
        .skip_while(|l| l.trim().is_empty())
        .take_while(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .collect();
    (title, para.join("\n"))
}

/// `Network registry — trained nets` → `network-registry-trained-nets`.
pub fn slug(text: &str) -> String {
    let mut s = String::new();
    for c in text.chars().flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            s.push(c);
        } else if !s.ends_with('-') && !s.is_empty() {
            s.push('-');
        }
    }
    s.trim_end_matches('-').to_string()
}

/// Markdown to HTML, with an id on every heading and wide tables in a scrolling box. Also returns
/// the level-2 and level-3 headings, `(level, id, text)`, for a contents list.
pub fn markdown_html(md: &str) -> (String, Vec<(u8, String, String)>) {
    let opts = Options::ENABLE_TABLES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_HEADING_ATTRIBUTES
        | Options::ENABLE_TASKLISTS;
    let mut events: Vec<Event> = Parser::new_ext(md, opts).collect();
    let mut toc = Vec::new();
    let mut used = std::collections::HashSet::new();
    let mut i = 0;
    while i < events.len() {
        if let Event::Start(Tag::Heading { level, id, .. }) = &events[i] {
            let level = *level;
            let mut text = String::new();
            let mut j = i + 1;
            while j < events.len() && !matches!(events[j], Event::End(TagEnd::Heading(_))) {
                if let Event::Text(t) | Event::Code(t) = &events[j] {
                    text.push_str(t);
                }
                j += 1;
            }
            let id = match id {
                Some(id) => id.to_string(),
                None => {
                    let base = match slug(&text) {
                        s if s.is_empty() => "section".to_string(),
                        s => s,
                    };
                    let mut id = base.clone();
                    let mut n = 2;
                    while used.contains(&id) {
                        id = format!("{base}-{n}");
                        n += 1;
                    }
                    id
                }
            };
            used.insert(id.clone());
            if let Event::Start(Tag::Heading { id: slot, .. }) = &mut events[i] {
                *slot = Some(CowStr::from(id.clone()));
            }
            let lv = match level {
                HeadingLevel::H1 => 1,
                HeadingLevel::H2 => 2,
                HeadingLevel::H3 => 3,
                _ => 4,
            };
            if lv == 2 || lv == 3 {
                toc.push((lv, id, text));
            }
            i = j;
        }
        i += 1;
    }
    let mut html = String::new();
    pulldown_cmark::html::push_html(&mut html, events.into_iter());
    let html = html
        .replace("<table>", "<div class=\"tw\"><table>")
        .replace("</table>", "</table></div>");
    (html, toc)
}

fn when(t: Option<SystemTime>) -> String {
    t.map(|t| {
        chrono::DateTime::<chrono::Local>::from(t)
            .format("%Y-%m-%d %H:%M")
            .to_string()
    })
    .unwrap_or_default()
}

fn page(title: &str, nav_extra: &[(&str, &str)], body: &str) -> String {
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>{}</title><style>{STYLE}{DOC_STYLE}</style></head><body>{}<main>{body}</main>{FILTER_JS}</body></html>",
        esc(title),
        nav("registry", nav_extra),
    )
}

/// The other registry files, as nav links.
fn file_links(docs: &[Doc], current: Option<&str>) -> String {
    let items: Vec<String> = docs
        .iter()
        .map(|d| {
            let name = d.file.trim_end_matches(".md");
            if Some(d.file.as_str()) == current {
                format!("<b>{name}</b>")
            } else {
                format!("<a href=\"{}\">{name}</a>", esc(&d.file))
            }
        })
        .collect();
    format!("<p class=\"meta\">{}</p>", items.join(" · "))
}

/// `GET /registry/`: every file, its title, its opening paragraph and when it last changed.
pub fn render_index(dir: &Path) -> String {
    let docs = list(dir);
    if docs.is_empty() {
        return page(
            "registry",
            &[],
            &format!(
                "<h1>registry</h1><p>No registry files in <code>{}</code> (<code>serve --registry-dir</code>).</p>",
                esc(&dir.display().to_string())
            ),
        );
    }
    let mut body = String::from(
        "<h1>registry</h1><p class=\"meta\">What exists and how it was made: corpora, nets, \
         experiments, matches. Read from the repo on every request.</p>",
    );
    for d in &docs {
        let (summary, _) = markdown_html(&d.summary);
        body.push_str(&format!(
            "<h2><a href=\"{file}\">{title}</a></h2><p class=\"meta\">{file} · updated {when} · \
             <a href=\"raw/{file}\">raw</a></p>{summary}",
            file = esc(&d.file),
            title = esc(&d.title),
            when = when(d.modified),
        ));
    }
    page("registry", &[], &body)
}

/// `GET /registry/<file>`: one file rendered, with the other files and a contents list on top.
/// `None` when there is no such file.
pub fn render_doc(dir: &Path, file: &str) -> Option<String> {
    let md = read(dir, file)?;
    let docs = list(dir);
    let doc = docs.iter().find(|d| d.file == file);
    let (html, toc) = markdown_html(&md);
    let mut body = file_links(&docs, Some(file));
    body.push_str(&format!(
        "<p class=\"meta\">{} · updated {} · <a href=\"raw/{}\">raw</a></p>",
        esc(file),
        when(doc.and_then(|d| d.modified)),
        esc(file)
    ));
    body.push_str(
        "<p><input id=\"q\" type=\"search\" placeholder=\"filter rows and items…\" autocomplete=\"off\"> \
         <span id=\"qn\" class=\"meta\"></span></p>",
    );
    if toc.len() > 1 {
        body.push_str(&format!("<details><summary>contents ({})</summary><ul>", toc.len()));
        for (level, id, text) in &toc {
            let pad = if *level == 3 {
                " style=\"margin-left:1.2em\""
            } else {
                ""
            };
            body.push_str(&format!("<li{pad}><a href=\"#{}\">{}</a></li>", esc(id), esc(text)));
        }
        body.push_str("</ul></details>");
    }
    body.push_str(&html);
    let title = doc.map_or(file.to_string(), |d| d.title.clone());
    Some(page(&title, &[], &body))
}

/// The file's text, if `file` names one in the registry.
pub fn read(dir: &Path, file: &str) -> Option<String> {
    if !valid_name(file) {
        return None;
    }
    std::fs::read_to_string(dir.join(file)).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_render_and_headings_get_ids() {
        let md = "# Title\n\nIntro [NETS.md](NETS.md).\n\n## v9 rules (2026-10-06 on)\n\n\
                  ### <a id=\"exp069\"></a>exp069 — policy targets\n\n\
                  | a | b |\n|---|---|\n| 1 | **2** |\n\n## v9 rules (2026-10-06 on)\n";
        let (html, toc) = markdown_html(md);
        assert!(html.contains("<div class=\"tw\"><table>"), "{html}");
        assert!(html.contains("<td><strong>2</strong></td>"), "{html}");
        assert!(html.contains("<a href=\"NETS.md\">NETS.md</a>"), "{html}");
        assert!(html.contains("<h2 id=\"v9-rules-2026-10-06-on\">"), "{html}");
        // The inline anchor survives, and the heading gets an id of its own.
        assert!(html.contains("<a id=\"exp069\"></a>"), "{html}");
        assert!(html.contains("<h3 id=\"exp069-policy-targets\">"), "{html}");
        // A repeated heading gets a distinct id.
        assert!(html.contains("<h2 id=\"v9-rules-2026-10-06-on-2\">"), "{html}");
        assert_eq!(toc.len(), 3);
        assert_eq!(toc[1].0, 3);
    }

    #[test]
    fn only_markdown_files_in_the_directory_are_opened() {
        assert!(valid_name("DATA.md"));
        assert!(valid_name("exp-069_notes.md"));
        assert!(!valid_name("../CLAUDE.md"));
        assert!(!valid_name("a/b.md"));
        assert!(!valid_name(".hidden.md"));
        assert!(!valid_name("DATA.txt"));
        assert!(!valid_name(".md"));
    }

    #[test]
    fn the_index_takes_the_title_and_the_first_paragraph() {
        let (t, s) = title_and_summary("# Data registry — corpora\n\nWhat was generated,\nby which net.\n\nMore.\n");
        assert_eq!(t.as_deref(), Some("Data registry — corpora"));
        assert_eq!(s, "What was generated,\nby which net.");
    }
}
