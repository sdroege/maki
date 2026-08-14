use super::segment::SegmentCache;
use crate::selection::{self, LineBreaks, ScreenSelection, Selection};
use crate::theme;

use maki_markdown::render::{CODE_BAR, CODE_BAR_WRAP};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::{Paragraph, Widget, Wrap};
use unicode_width::UnicodeWidthStr;

/// Per-row mapping from rendered cells back to raw text positions.
enum RowSourceMap {
    /// Row matched via subsequence matching. Contains per-char mapping.
    Matched {
        /// (screen_col, rendered_char_idx, raw_char_idx) for each matched
        /// content char.
        matches: Vec<(u16, usize, usize)>,
        /// Raw char index where leading skipped syntax starts.
        leading_skip_start: usize,
        /// Raw char index after the last matched char.
        raw_end: usize,
        /// True for table content rows.
        is_table_row: bool,
    },
    /// Row matched via pattern matching (HR, table separator).
    PatternMatched {
        /// Raw char range [start, end) for this row.
        raw_range: (usize, usize),
    },
    /// Row is synthetic (no source mapping).
    Synthetic,
}

/// Scrapes rendered chars from a buffer row, returning (screen_col, char) pairs.
/// Skips wide-char continuation cells. Starts at `col_start`.
fn scrape_row_chars(buf: &Buffer, row: u16, col_start: u16, col_end: u16) -> Vec<(u16, char)> {
    let mut chars = Vec::new();
    let mut skip_next: usize = 0;
    for col in col_start..=col_end {
        if skip_next > 0 {
            skip_next -= 1;
            continue;
        }
        let cell = &buf[(col, row)];
        let sym = cell.symbol();
        for ch in sym.chars() {
            chars.push((col, ch));
        }
        skip_next = UnicodeWidthStr::width(sym).saturating_sub(1);
    }
    chars
}

/// Detects and strips a synthetic prefix from the rendered chars. Returns the
/// content chars (after prefix) and whether the row is entirely synthetic.
///
/// Returns (content_chars, is_entirely_synthetic, is_table_row).
/// content_chars is a Vec<(screen_col, char)> with the prefix stripped.
fn strip_synthetic_prefix(
    buf: &Buffer,
    row: u16,
    col_start: u16,
    col_end: u16,
) -> (Vec<(u16, char)>, bool, bool) {
    let first_cell = &buf[(col_start, row)];
    let first_sym = first_cell.symbol();

    // Code bar: │ with code_gutter style. First rows of a code line use "│ "
    // (2 cols); wrapped continuation rows use "│" (1 col, content at col 1).
    if first_sym == "│" && first_cell.style().fg == theme::current().code_gutter.fg {
        let strip_cols = if (col_start + 1) <= col_end && buf[(col_start + 1, row)].symbol() == " "
        {
            CODE_BAR.width() as u16
        } else {
            CODE_BAR_WRAP.width() as u16
        };
        let chars = scrape_row_chars(buf, row, col_start + strip_cols, col_end);
        return (chars, false, false);
    }

    // Table border row (separator ├, top/bottom ╭╰, HR ─, empty cells):
    // entirely synthetic
    if is_table_border_row(buf, row, col_start, col_end) {
        return (Vec::new(), true, false);
    }

    // Table content row: │ with table_border style — split by │ borders,
    // trim each cell
    if first_sym == "│" && first_cell.style().fg == theme::current().table_border.fg {
        let chars = scrape_table_content(buf, row, col_start, col_end);
        return (chars, false, true);
    }

    // List marker: • or <digits>. — may be indented (nested list)
    let marker_col = (col_start..=col_end)
        .find(|&c| buf[(c, row)].symbol() != " ")
        .unwrap_or(col_end + 1);
    if marker_col <= col_end {
        let marker = buf[(marker_col, row)].symbol();
        // Bullet
        if marker == "•" {
            let chars = scrape_row_chars(buf, row, marker_col + 2, col_end);
            return (chars, false, false);
        }
        // Ordered list marker: digit(s) followed by ". "
        if marker.len() == 1 && marker.chars().next().unwrap().is_ascii_digit() {
            let mut col = marker_col;
            let mut digit_end = marker_col;
            while col <= col_end {
                let c = buf[(col, row)].symbol();
                if c.len() == 1 && c.chars().next().unwrap().is_ascii_digit() {
                    digit_end = col + 1;
                    col += 1;
                } else {
                    break;
                }
            }
            if digit_end <= col_end
                && buf[(digit_end, row)].symbol() == "."
                && digit_end < col_end
                && buf[(digit_end + 1, row)].symbol() == " "
            {
                let chars = scrape_row_chars(buf, row, digit_end + 2, col_end);
                return (chars, false, false);
            }
        }
    }

    // No synthetic prefix detected
    let chars = scrape_row_chars(buf, row, col_start, col_end);
    (chars, false, false)
}

/// Checks if a row is entirely table border chars.
fn is_table_border_row(buf: &Buffer, row: u16, col_start: u16, col_end: u16) -> bool {
    let border_chars = "│╭╮├┤┼╰╯─┬┴";
    (col_start..=col_end).all(|c| {
        let sym = buf[(c, row)].symbol();
        sym == " " || border_chars.contains(sym.chars().next().unwrap_or(' '))
    })
}

/// Scrapes table content by splitting on │ borders and trimming each cell.
fn scrape_table_content(buf: &Buffer, row: u16, col_start: u16, col_end: u16) -> Vec<(u16, char)> {
    let mut chars = Vec::new();
    let mut in_cell = false;

    for col in col_start..=col_end {
        let cell = &buf[(col, row)];
        if cell.symbol() == "│" && cell.style().fg == theme::current().table_border.fg {
            // End of cell — trim trailing spaces from collected chars
            if in_cell {
                while chars.last().is_some_and(|(_, c)| *c == ' ') {
                    chars.pop();
                }
            }
            in_cell = false;
        } else if cell.symbol() != " " {
            // Start of cell content
            if !in_cell {
                in_cell = true;
            }
            for ch in cell.symbol().chars() {
                chars.push((col, ch));
            }
        }
    }
    // Trim trailing spaces from last cell
    while chars.last().is_some_and(|(_, c)| *c == ' ') {
        chars.pop();
    }
    chars
}

/// Greedily matches rendered chars against raw chars starting at raw_pos.
/// Returns (matches, new_raw_pos). Each match is (screen_col, rendered_idx, raw_idx).
/// All-or-nothing: if any rendered char fails to match, returns empty matches
/// and the original raw_pos (no advancement).
fn match_subsequence(
    raw: &[char],
    raw_pos: usize,
    rendered: &[(u16, char)],
) -> (Vec<(u16, usize, usize)>, usize) {
    if rendered.is_empty() {
        return (Vec::new(), raw_pos);
    }
    let orig_raw_pos = raw_pos;
    let mut matches = Vec::with_capacity(rendered.len());
    let mut rp = raw_pos;

    for (ri, &(screen_col, ch)) in rendered.iter().enumerate() {
        let mut found = false;
        while rp < raw.len() {
            if raw[rp] == ch {
                matches.push((screen_col, ri, rp));
                rp += 1;
                found = true;
                break;
            }
            rp += 1;
        }
        if !found {
            return (Vec::new(), orig_raw_pos);
        }
    }
    (matches, rp)
}

/// Scans forward from `pos` looking for known closing syntax tokens.
/// Returns the position after the last included token, or `pos` if none found.
/// Stops at paragraph breaks (\n\n) or unknown chars.
fn scan_closing_syntax(raw: &[char], pos: usize, bound: usize) -> usize {
    let mut end = pos;
    let mut i = pos;
    while i < bound {
        // Skip newlines
        while i < bound && raw[i] == '\n' {
            i += 1;
        }
        // Check for closing syntax token
        let remaining = &raw[i..bound];
        let token_len = if remaining.starts_with(&['`', '`', '`']) {
            3
        } else if remaining.starts_with(&['*', '*'])
            || remaining.starts_with(&['_', '_'])
            || remaining.starts_with(&['~', '~'])
        {
            2
        } else if matches!(remaining.first(), Some(&'`') | Some(&'*') | Some(&'_')) {
            1
        } else {
            break;
        };
        end = i + token_len;
        i += token_len;
    }
    end
}

/// Scans forward from `pos` looking for known opening syntax tokens.
/// Returns the position after the last included token, or `pos` if none found.
fn scan_opening_syntax(raw: &[char], pos: usize, bound: usize) -> usize {
    let mut end = pos;
    let mut i = pos;
    while i < bound {
        // Skip newlines
        while i < bound && raw[i] == '\n' {
            i += 1;
        }
        // Skip indent spaces of a line-leading marker (nested list)
        if i > pos && raw[i - 1] == '\n' {
            while i < bound && raw[i] == ' ' {
                i += 1;
            }
        }
        let remaining = &raw[i..bound];
        let token_len = if remaining.starts_with(&['`', '`', '`']) {
            3
        } else if remaining.starts_with(&['*', '*'])
            || remaining.starts_with(&['_', '_'])
            || remaining.starts_with(&['~', '~'])
        {
            2
        } else if remaining.first() == Some(&'`') {
            1
        } else if (remaining.first() == Some(&'#')
            || matches!(remaining.first(), Some(&'-') | Some(&'*') | Some(&'+')))
            && remaining.get(1) == Some(&' ')
        {
            2
        } else if matches!(remaining.first(), Some(&'*') | Some(&'_')) {
            1
        } else if remaining.first().is_some_and(|c| c.is_ascii_digit()) {
            // Ordered list: digits followed by ". "
            let mut j = i;
            while j < bound && raw[j].is_ascii_digit() {
                j += 1;
            }
            if j + 2 <= bound && raw[j] == '.' && raw[j + 1] == ' ' {
                (j - i) + 2
            } else {
                break;
            }
        } else {
            break;
        };
        end = i + token_len;
        i += token_len;
    }
    end
}

/// Searches forward from `pos` for a full-line horizontal rule matching
/// `^([-*_])\1{2,}$` (3+ of the same char, nothing else on the line).
/// Returns the raw range (start, end) if found, or None.
fn find_hr_pattern(raw: &[char], pos: usize) -> Option<(usize, usize)> {
    let remaining = &raw[pos..];
    // `pos` can land mid-line (e.g. after a table row's unmatched tail), so
    // index 0 of `remaining` is only a line start when `pos` is one.
    let pos_at_line_start = pos == 0 || raw[pos - 1] == '\n';
    for line_start in 0..remaining.len().saturating_sub(2) {
        let at_line_start = if line_start == 0 {
            pos_at_line_start
        } else {
            remaining[line_start - 1] == '\n'
        };
        if !at_line_start {
            continue;
        }
        let ch = remaining[line_start];
        if !matches!(ch, '-' | '*' | '_') {
            continue;
        }
        let mut line_end = line_start + 1;
        while line_end < remaining.len() && remaining[line_end] == ch {
            line_end += 1;
        }
        let at_line_end = line_end == remaining.len() || remaining[line_end] == '\n';
        if line_end - line_start >= 3 && at_line_end {
            let abs_start = pos + line_start;
            let abs_end = pos + line_end;
            return Some((abs_start, abs_end));
        }
    }
    None
}

/// Searches forward from `pos` for a full-line table separator matching
/// `^\|[-: |]+\|$` (e.g. |---|---|). Returns the raw range (start, end)
/// if found, or None.
fn find_table_sep_pattern(raw: &[char], pos: usize) -> Option<(usize, usize)> {
    let remaining = &raw[pos..];
    // `pos` can land mid-line (e.g. after a table row's unmatched tail), so
    // index 0 of `remaining` is only a line start when `pos` is one.
    let pos_at_line_start = pos == 0 || raw[pos - 1] == '\n';
    for line_start in 0..remaining.len().saturating_sub(2) {
        let at_line_start = if line_start == 0 {
            pos_at_line_start
        } else {
            remaining[line_start - 1] == '\n'
        };
        if !at_line_start {
            continue;
        }
        if remaining[line_start] != '|' {
            continue;
        }
        let mut line_end = line_start + 1;
        while line_end < remaining.len()
            && remaining[line_end] != '\n'
            && (remaining[line_end].is_whitespace()
                || matches!(remaining[line_end], '-' | ':' | '|'))
        {
            line_end += 1;
        }
        // The line must end here: nothing but trailing whitespace after the
        // final `|`, then `\n` or end of slice.
        if line_end < remaining.len() && remaining[line_end] != '\n' {
            continue;
        }
        // Trim trailing whitespace so the range ends at the final `|`.
        let mut end = line_end;
        while end > line_start + 1 && remaining[end - 1].is_whitespace() {
            end -= 1;
        }
        if end > line_start + 1
            && remaining[end - 1] == '|'
            && remaining[line_start + 1..end].contains(&'-')
        {
            return Some((pos + line_start, pos + end));
        }
    }
    None
}

/// Builds per-row source maps by matching rendered buffer content against raw
/// text. Returns one `RowSourceMap` per rendered row.
fn build_row_source_maps(
    buf: &Buffer,
    area: Rect,
    prefix_width: u16,
    raw_chars: &[char],
) -> Vec<RowSourceMap> {
    let height = area.height as usize;
    let mut maps = Vec::with_capacity(height);
    let mut raw_pos = 0usize;

    for row_idx in 0..height {
        let row = area.y + row_idx as u16;
        let col_start = if row_idx == 0 { prefix_width } else { area.x };
        let col_end = area.x + area.width.saturating_sub(1);

        // Check if row is blank
        let is_blank = (col_start..=col_end).all(|c| buf[(c, row)].symbol() == " ");
        if is_blank {
            maps.push(RowSourceMap::Synthetic);
            continue;
        }

        let (mut content, is_entirely_synthetic, is_table_row) =
            strip_synthetic_prefix(buf, row, col_start, col_end);

        // Entirely synthetic rows: pattern-match HR (─) and table separator
        // (├) rows. Top/bottom borders (╭/╰) have no raw counterpart.
        if is_entirely_synthetic {
            let first = (col_start..=col_end)
                .find(|&c| buf[(c, row)].symbol() != " ")
                .map(|c| buf[(c, row)].symbol());
            let range = match first {
                Some("─") => find_hr_pattern(raw_chars, raw_pos),
                Some("├") => find_table_sep_pattern(raw_chars, raw_pos),
                _ => None,
            };
            if let Some(range) = range {
                raw_pos = range.1 + 1;
                maps.push(RowSourceMap::PatternMatched { raw_range: range });
            } else {
                maps.push(RowSourceMap::Synthetic);
            }
            continue;
        }

        // Trim trailing spaces from rendered content (buffer padding).
        while content.last().is_some_and(|(_, c)| *c == ' ') {
            content.pop();
        }
        // Empty content after stripping prefix (e.g. only whitespace)
        if content.is_empty() {
            maps.push(RowSourceMap::Synthetic);
            continue;
        }

        // Greedy subsequence match
        let leading_skip_start = raw_pos;
        let (matches, new_raw_pos) = match_subsequence(raw_chars, raw_pos, &content);

        if matches.is_empty() && !content.is_empty() {
            maps.push(RowSourceMap::Synthetic);
            continue;
        }

        // Consume this row's trailing closing syntax (same raw line) so the
        // next row's leading_skip_start doesn't point into it.
        let line_end = raw_chars[new_raw_pos..]
            .iter()
            .position(|&c| c == '\n')
            .map(|i| new_raw_pos + i)
            .unwrap_or(raw_chars.len());
        raw_pos = scan_closing_syntax(raw_chars, new_raw_pos, line_end);
        maps.push(RowSourceMap::Matched {
            matches,
            leading_skip_start,
            raw_end: new_raw_pos,
            is_table_row,
        });
    }
    maps
}

/// Computes the upper bound for trailing extension: the next non-synthetic
/// row's leading position, or raw_chars.len().
fn next_row_bound(maps: &[RowSourceMap], row_idx: usize, raw_len: usize) -> usize {
    for m in maps.iter().skip(row_idx + 1) {
        match m {
            RowSourceMap::Matched {
                leading_skip_start, ..
            } => return *leading_skip_start,
            RowSourceMap::PatternMatched { raw_range } => return raw_range.0,
            RowSourceMap::Synthetic => continue,
        }
    }
    raw_len
}

/// End of the raw range a selected `Matched` row claims, after extension:
/// end of the raw line for table rows, past trailing closing syntax otherwise.
/// Closing syntax is only included within the row's own raw line — unrendered
/// lines like code fences never leak into a code row's claim.
fn matched_claim_end(
    raw_end: usize,
    is_table_row: bool,
    maps: &[RowSourceMap],
    row_idx: usize,
    raw_chars: &[char],
) -> usize {
    if is_table_row {
        raw_chars[raw_end..]
            .iter()
            .position(|&c| c == '\n')
            .map(|i| raw_end + i)
            .unwrap_or(raw_chars.len())
    } else {
        let line_end = raw_chars[raw_end..]
            .iter()
            .position(|&c| c == '\n')
            .map(|i| raw_end + i)
            .unwrap_or(raw_chars.len());
        let bound = next_row_bound(maps, row_idx, raw_chars.len()).min(line_end);
        scan_closing_syntax(raw_chars, raw_end, bound)
    }
}

/// Emits the inter-line part of a gap. Gaps crossing a raw-line boundary
/// contain unrendered raw (code fences) — only the trailing whitespace run
/// (wrap/paragraph newlines) survives. Same-line gaps (wrap points) are
/// emitted verbatim.
fn emit_inter_gap(out: &mut String, gap: &[char], first_non_blank: bool) {
    if !gap.contains(&'\n') {
        emit_gap(out, gap, first_non_blank);
        return;
    }
    let keep_from = gap
        .iter()
        .rposition(|c| !c.is_whitespace())
        .map(|i| i + 1)
        .unwrap_or(0);
    emit_gap(out, &gap[keep_from..], first_non_blank);
}

/// Emits the gap between the previous row's raw end and the current row's raw
/// start. Wrap-point gaps (spaces/single newlines) become `\n`. Paragraph
/// breaks (`\n\n`) and non-whitespace content (leading syntax) are emitted
/// verbatim.
fn emit_gap(out: &mut String, gap: &[char], first_non_blank: bool) {
    if gap.is_empty() {
        if !first_non_blank {
            out.push('\n');
        }
        return;
    }
    // Non-whitespace gap (leading syntax like `**`, `` ` ``) — emit verbatim
    if gap.iter().any(|&c| !c.is_whitespace()) {
        out.push_str(&gap.iter().collect::<String>());
        return;
    }
    // All-whitespace gap containing paragraph break — emit verbatim, but not
    // before the first non-blank row (that would include preceding content)
    let gap_str: String = gap.iter().collect();
    if gap_str.contains("\n\n") && !first_non_blank {
        out.push_str(&gap_str);
        return;
    }
    // Wrap point (spaces/newlines) — emit \n
    if !first_non_blank {
        out.push('\n');
    }
}

/// Checks if a row is fully selected given the selection's column range.
fn is_row_fully_selected(
    ss: &ScreenSelection,
    row: u16,
    left: u16,
    right: u16,
    maps: &[RowSourceMap],
    row_idx: usize,
) -> bool {
    let (sel_start, sel_end) = selection::col_range(ss, left, right, row);
    let map = maps.get(row_idx);
    match map {
        Some(RowSourceMap::Matched { matches, .. }) if !matches.is_empty() => {
            let first_col = matches.first().unwrap().0;
            let last_col = matches.last().unwrap().0;
            sel_start <= first_col && sel_end >= last_col
        }
        Some(RowSourceMap::PatternMatched { .. }) => sel_start <= left && sel_end >= right,
        _ => sel_start <= left && sel_end >= right,
    }
}

/// Extracts raw text for a selection using the source map.
fn extract_via_source_map(
    buf: &Buffer,
    area: Rect,
    ss: &ScreenSelection,
    raw_chars: &[char],
    prefix_width: u16,
    maps: &[RowSourceMap],
) -> String {
    let left = area.x;
    let right = area.x + area.width.saturating_sub(1);
    let mut out = String::new();
    let mut first_non_blank = true;
    // Initialize claimed_end past the raw ranges of unselected preceding rows,
    // so the first selected row's gap doesn't include their content.
    let mut claimed_end = 0usize;
    for (idx, m) in maps.iter().take(ss.start_row as usize).enumerate() {
        match m {
            RowSourceMap::Matched {
                raw_end,
                is_table_row,
                ..
            } => {
                claimed_end = matched_claim_end(*raw_end, *is_table_row, maps, idx, raw_chars);
            }
            RowSourceMap::PatternMatched { raw_range } => claimed_end = raw_range.1,
            RowSourceMap::Synthetic => {}
        }
    }

    for row_idx in ss.start_row as usize..=ss.end_row as usize {
        let row = area.y + row_idx as u16;
        let col_start = if row_idx == 0 { prefix_width } else { area.x };
        let col_end = area.x + area.width.saturating_sub(1);

        // Skip blank rows
        let is_blank = (col_start..=col_end).all(|c| buf[(c, row)].symbol() == " ");
        if is_blank {
            continue;
        }

        let map = maps.get(row_idx);
        let fully_selected = is_row_fully_selected(ss, row, left, right, maps, row_idx);

        match map {
            Some(RowSourceMap::Synthetic) => {
                if !first_non_blank {
                    out.push('\n');
                }
                let (sel_start, sel_end) = selection::col_range(ss, left, right, row);
                let mut skip_next: usize = 0;
                for col in sel_start..=sel_end {
                    if skip_next > 0 {
                        skip_next -= 1;
                        continue;
                    }
                    let sym = buf[(col, row)].symbol();
                    out.push_str(sym);
                    skip_next = UnicodeWidthStr::width(sym).saturating_sub(1);
                }
                while out.ends_with(' ') {
                    out.pop();
                }
                first_non_blank = false;
            }
            Some(RowSourceMap::PatternMatched { raw_range }) => {
                if fully_selected {
                    emit_inter_gap(
                        &mut out,
                        &raw_chars[claimed_end..raw_range.0],
                        first_non_blank,
                    );
                    out.push_str(
                        &raw_chars[raw_range.0..raw_range.1]
                            .iter()
                            .collect::<String>(),
                    );
                    claimed_end = raw_range.1;
                } else {
                    if !first_non_blank {
                        out.push('\n');
                    }
                    let (sel_start, sel_end) = selection::col_range(ss, left, right, row);
                    let mut skip_next: usize = 0;
                    for col in sel_start..=sel_end {
                        if skip_next > 0 {
                            skip_next -= 1;
                            continue;
                        }
                        let sym = buf[(col, row)].symbol();
                        out.push_str(sym);
                        skip_next = UnicodeWidthStr::width(sym).saturating_sub(1);
                    }
                    // Claim the full raw range so the unselected remainder
                    // can't leak into the next row's gap.
                    claimed_end = raw_range.1;
                }
                first_non_blank = false;
            }
            Some(RowSourceMap::Matched {
                matches,
                leading_skip_start,
                raw_end,
                is_table_row,
            }) => {
                if fully_selected {
                    let (raw_start, raw_end_ext, has_opening) = if *is_table_row {
                        // Full raw line extension
                        let first_match_raw = matches.first().unwrap().2;
                        let prev_nl = (0..first_match_raw)
                            .rev()
                            .find(|&i| raw_chars[i] == '\n')
                            .map(|i| i + 1)
                            .unwrap_or(0);
                        let rs = prev_nl.max(claimed_end);
                        let re = matched_claim_end(*raw_end, true, maps, row_idx, raw_chars);
                        (rs, re, false)
                    } else {
                        // Leading extension
                        let ext_start = (*leading_skip_start).max(claimed_end);
                        let scanned = scan_opening_syntax(raw_chars, ext_start, *raw_end);
                        let has_opening = scanned > ext_start;
                        let raw_start = if has_opening {
                            scanned
                        } else {
                            matches.first().unwrap().2
                        };
                        let raw_end_ext =
                            matched_claim_end(*raw_end, false, maps, row_idx, raw_chars);
                        (raw_start, raw_end_ext, has_opening)
                    };

                    // Split the gap at this row's raw-line start: unrendered
                    // inter-line raw (code fences) is dropped, the row's own
                    // leading syntax is emitted verbatim. Without opening
                    // tokens the whole gap is inter-line.
                    let first_match_raw = matches.first().unwrap().2;
                    let line_start_raw = (0..first_match_raw)
                        .rev()
                        .find(|&i| raw_chars[i] == '\n')
                        .map(|i| i + 1)
                        .unwrap_or(0);
                    let raw_start = raw_start.max(line_start_raw);
                    let inter_end = if has_opening {
                        line_start_raw.max(claimed_end).min(raw_start)
                    } else {
                        raw_start
                    };
                    emit_inter_gap(
                        &mut out,
                        &raw_chars[claimed_end..inter_end],
                        first_non_blank,
                    );
                    let intra = &raw_chars[inter_end..raw_start];
                    if !intra.is_empty() {
                        out.push_str(&intra.iter().collect::<String>());
                    }
                    out.push_str(&raw_chars[raw_start..raw_end_ext].iter().collect::<String>());
                    claimed_end = raw_end_ext;
                } else {
                    // Partial: map columns to raw positions
                    let (sel_start, sel_end) = selection::col_range(ss, left, right, row);
                    let mut first_match: Option<usize> = None;
                    let mut last_match: Option<usize> = None;
                    for &(sc, _, ri) in matches {
                        if sc >= sel_start && sc <= sel_end {
                            first_match.get_or_insert(ri);
                            last_match = Some(ri);
                        }
                    }
                    if let (Some(fs), Some(ls)) = (first_match, last_match) {
                        // Partial selection: only emit \n between rows, no content gap
                        if !first_non_blank {
                            out.push('\n');
                        }
                        out.push_str(&raw_chars[fs..=ls].iter().collect::<String>());
                        // Claim the row's full extended range so the next row's
                        // gap can't leak this row's unselected closing syntax.
                        claimed_end =
                            matched_claim_end(*raw_end, *is_table_row, maps, row_idx, raw_chars);
                    }
                }
                first_non_blank = false;
            }
            _ => {
                // Fallback to rendered
                if !first_non_blank {
                    out.push('\n');
                }
                let (sel_start, sel_end) = selection::col_range(ss, left, right, row);
                let mut skip_next: usize = 0;
                for col in sel_start..=sel_end {
                    if skip_next > 0 {
                        skip_next -= 1;
                        continue;
                    }
                    let sym = buf[(col, row)].symbol();
                    out.push_str(sym);
                    skip_next = UnicodeWidthStr::width(sym).saturating_sub(1);
                }
                first_non_blank = false;
            }
        }
    }
    out
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

        // Source-map path: copy_markdown with partial selection on a valid segment
        if copy_markdown
            && seg.has_source_map
            && let Some(raw) = &seg.raw_text
            && seg.prefix_width < width
        {
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

            let raw_chars: Vec<char> = raw.chars().collect();
            let maps = build_row_source_maps(&tmp, tmp_area, seg.prefix_width, &raw_chars);

            let text =
                extract_via_source_map(&tmp, tmp_area, &ss, &raw_chars, seg.prefix_width, &maps);
            out.push_str(&text);
            continue;
        }

        // Fallback: rendered text extraction
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

#[cfg(test)]
mod pattern_match_tests {
    use super::{find_hr_pattern, find_table_sep_pattern};
    use test_case::test_case;

    #[test_case("before\n---\nafter", 0, Some((7, 10)); "full_line_dash_hr")]
    #[test_case("before\n***\nafter", 0, Some((7, 10)); "full_line_star_hr")]
    #[test_case("a --- b\n---\n", 0, Some((8, 11)); "mid_line_run_skipped")]
    #[test_case("x --- y\n***\n", 6, Some((8, 11)); "pos_mid_line")]
    #[test_case("a --- b\n", 0, None; "mid_line_run_only")]
    #[test_case("--- x\n", 0, None; "trailing_content_after_run")]
    #[test_case("--\n", 0, None; "run_too_short")]
    fn hr_pattern_full_line_only(raw: &str, pos: usize, expected: Option<(usize, usize)>) {
        let chars: Vec<char> = raw.chars().collect();
        assert_eq!(find_hr_pattern(&chars, pos), expected);
    }

    #[test_case("| a |\n|---|---|\n| b |", 0, Some((6, 15)); "full_line")]
    #[test_case("x |---|---|\n|---|---|\n", 0, Some((12, 21)); "mid_line_start_skipped")]
    #[test_case("|---|---| extra\n", 0, None; "trailing_content")]
    #[test_case("x |---|---|\n", 0, None; "mid_line_only")]
    #[test_case("| a |\n|---|---|\n", 4, Some((6, 15)); "pos_mid_line")]
    #[test_case("| a |\n|---|---|", 0, Some((6, 15)); "no_trailing_newline")]
    #[test_case("|:---:|\n", 0, Some((0, 7)); "alignment_colons")]
    fn table_sep_full_line_only(raw: &str, pos: usize, expected: Option<(usize, usize)>) {
        let chars: Vec<char> = raw.chars().collect();
        assert_eq!(find_table_sep_pattern(&chars, pos), expected);
    }
}
