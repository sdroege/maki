use super::segment::SegmentCache;
use crate::selection::{self, LineBreaks, ScreenSelection, Selection};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::{Paragraph, Widget, Wrap};
use unicode_width::UnicodeWidthChar;

/// Simulates wrapping for a single source line at the given width, returning
/// the (col_start, col_end) char offsets for each rendered row it produces.
fn wrap_line_offsets(text: &str, width: usize) -> Vec<(usize, usize)> {
    if width == 0 || text.is_empty() {
        return vec![(0, text.chars().count())];
    }
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if len == 0 {
        return vec![(0, 0)];
    }

    let mut offsets = Vec::new();
    let mut col = 0usize;
    let mut line_start = 0;
    let mut last_breakable: Option<usize> = None;
    let mut i = 0;

    loop {
        if i >= len {
            offsets.push((line_start, len));
            break;
        }

        let cw = chars[i].width().unwrap_or(0);

        if chars[i] == ' ' || chars[i] == '\t' {
            last_breakable = Some(i);
        }

        if col + cw > width && col > 0 {
            if let Some(bp) = last_breakable {
                offsets.push((line_start, bp));
                line_start = bp + 1;
                while line_start < len && chars[line_start] == ' ' {
                    line_start += 1;
                }
            } else {
                offsets.push((line_start, i));
                line_start = i;
            }
            col = 0;
            last_breakable = None;
            // Recalculate col from line_start to current i
            for ch in chars.iter().take(i).skip(line_start) {
                col += ch.width().unwrap_or(0);
            }
            continue;
        }

        col += cw;
        i += 1;
    }

    offsets
}

/// Builds a raw-to-rendered row mapping for a segment's source lines.
/// Each source line is wrapped independently at `wrap_width`, producing
/// a sequence of (source_line_idx, col_start, col_end) entries.
fn build_raw_row_map(lines: &[&str], wrap_width: u16) -> Vec<(usize, usize, usize)> {
    if wrap_width == 0 || lines.is_empty() {
        return Vec::new();
    }
    let w = wrap_width as usize;
    let mut rows = Vec::new();

    for (idx, &text) in lines.iter().enumerate() {
        let offsets = wrap_line_offsets(text, w);
        for (cs, ce) in offsets {
            rows.push((idx, cs, ce));
        }
    }

    rows
}

pub(super) fn extract_selection_text(
    cache: &SegmentCache,
    viewport_width: u16,
    sel: &Selection,
    msg_area: Rect,
    copy_markdown: bool,
) -> String {
    let (doc_start, doc_end) = sel.normalized();
    let width = viewport_width;

    let heights: Vec<u16> = cache.segments().iter().map(|s| s.height(width)).collect();

    let mut out = String::new();
    let mut doc_row: u32 = 0;

    for (i, &h) in heights.iter().enumerate() {
        let seg_start = doc_row;
        let seg_end = doc_row + h as u32;
        doc_row = seg_end;

        if seg_end <= doc_start.row || seg_start > doc_end.row {
            continue;
        }

        if !out.is_empty() {
            out.push('\n');
        }

        let Some(seg) = cache.get(i) else { continue };

        if seg.lines().is_empty() {
            continue;
        }

        let seg_fully_selected = seg_start >= doc_start.row
            && seg_end <= doc_end.row + 1
            && doc_start.col <= msg_area.x
            && doc_end.col >= msg_area.x + width - 1;
        if seg_fully_selected
            && copy_markdown
            && let Some(raw) = &seg.raw_text
        {
            out.push_str(raw);
            continue;
        }

        // Partial selection with raw text available: extract raw text for selected rows
        if copy_markdown
            && let Some(raw) = &seg.raw_text
            && seg.prefix_width < width
        {
            let rel_start = doc_start.row.saturating_sub(seg_start) as usize;
            let rel_end = ((doc_end.row + 1).saturating_sub(seg_start) as usize).min(h as usize);
            let wrap_width = width - seg.prefix_width;
            // Build row map from raw text (not rendered lines with prefix) so char
            // positions match what we extract from.
            let raw_lines: Vec<&str> = raw.lines().collect();
            let raw_row_map = build_raw_row_map(&raw_lines, wrap_width);

            let to_col = |c: u16| {
                c.saturating_sub(msg_area.x)
                    .saturating_sub(seg.prefix_width) as usize
            };
            let start_col = if seg_start == doc_start.row {
                to_col(doc_start.col)
            } else {
                0
            };
            let end_col = if seg_end == doc_end.row + 1 {
                to_col(doc_end.col) + 1
            } else {
                wrap_width as usize
            };

            let first_row = raw_row_map.get(rel_start);
            let last_row = raw_row_map.get(rel_end.saturating_sub(1));
            if let (Some(&(_, _, first_ce)), Some(&(_, _, _))) = (first_row, last_row) {
                for (idx, &(src, cs, ce)) in raw_row_map[rel_start..rel_end].iter().enumerate() {
                    if idx > 0 {
                        out.push('\n');
                    }
                    let line = raw_lines[src];
                    let chars: Vec<char> = line.chars().collect();
                    let (s, e) = if idx == 0 {
                        (cs + start_col, (cs + end_col).min(first_ce))
                    } else if idx == rel_end - rel_start - 1 {
                        (cs, (cs + end_col).min(chars.len()))
                    } else {
                        (cs, ce)
                    };
                    let s = s.min(chars.len());
                    let e = e.min(chars.len());
                    if s < e {
                        out.push_str(chars[s..e].iter().collect::<String>().as_str());
                    }
                }
                continue;
            }
        }

        let tmp_area = Rect::new(0, 0, width, h);
        let mut tmp = Buffer::empty(tmp_area);
        Paragraph::new(seg.lines().to_vec())
            .wrap(Wrap { trim: false })
            .render(tmp_area, &mut tmp);

        let rel_start = doc_start.row.saturating_sub(seg_start) as u16;
        let rel_end = ((doc_end.row + 1).saturating_sub(seg_start) as u16).min(h);

        let start_col = if seg_start > doc_start.row {
            0
        } else {
            doc_start.col.saturating_sub(msg_area.x)
        };
        let end_col = if seg_end < doc_end.row + 1 {
            width.saturating_sub(1)
        } else {
            doc_end.col.saturating_sub(msg_area.x)
        };

        let ss = ScreenSelection {
            start_row: rel_start,
            start_col,
            end_row: rel_end.saturating_sub(1),
            end_col,
        };

        let breaks = LineBreaks::from_lines(seg.lines(), width);
        selection::append_rows(&tmp, tmp_area, &ss, rel_start, rel_end, &mut out, &breaks);
    }
    out
}
