//! Embedding Helix in a host application (feature `embed`): after every
//! frame the host gets the overlays it draws itself, as data. Overlays the
//! host draws are left out of the cell frames; Helix keeps handling their
//! keys. The host runs commands in Helix's event loop ([`HostRequest`]),
//! and Helix asks the host for its own UI ([`HostAction`]).

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use helix_core::command_line::Args;
use helix_core::doc_formatter::DocumentFormatter;
use helix_core::Position;
use helix_view::tree::{HostLayout, HostLayoutEvent};
use helix_view::{graphics::Rect, Editor, View, ViewId};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::commands::MappableCommand;
use crate::compositor::{self, Component, Compositor, Event};
use crate::job::Jobs;
use crate::ui::lsp::{hover::Hover, signature_help::SignatureHelp};
use crate::ui::PromptEvent;
use crate::ui::{picker, EditorView, Markdown, Popup, Prompt};

/// A prompt layer: the command line (`:`), search (`/`), and the like.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptState {
    /// What the prompt asks, e.g. `:` or `search:`.
    pub prefix: String,
    pub line: String,
    /// The cursor, in bytes of `line`.
    pub cursor: usize,
    pub completions: Vec<String>,
    pub selection: Option<usize>,
}

/// The completion menu of insert mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionState {
    /// A window of the matching items, starting at match `offset`.
    pub items: Vec<CompletionItemState>,
    pub offset: usize,
    /// All matching items.
    pub total: usize,
    /// The selected match, if any.
    pub selection: Option<usize>,
    /// The word typed so far, which the items are filtered by.
    pub filter: String,
    /// The selected item's detail and documentation, as markdown.
    pub documentation: Option<String>,
    /// The cursor the menu belongs to, in cells of the screen.
    pub anchor: Position,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionItemState {
    pub label: String,
    /// `function`, `variable`, … (LSP) or the source (`word`, `path`).
    pub kind: String,
    pub detail: Option<String>,
    pub deprecated: bool,
}

/// The info box: which-key menus (space mode, `g`, …) and register lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfoState {
    pub title: String,
    /// One `key  description` row per line, keys padded to one width.
    pub text: String,
}

/// A popup with markdown (hover and the like).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PopupState {
    pub id: &'static str,
    pub markdown: String,
    /// The text the popup refers to, in cells of the screen.
    pub anchor: Position,
}

/// Signature help while typing a call's arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureState {
    pub signature: String,
    /// The active parameter, in bytes of `signature`.
    pub active_parameter: Option<(usize, usize)>,
    pub documentation: Option<String>,
    /// Which of `count` signatures is shown.
    pub index: usize,
    pub count: usize,
    pub anchor: Position,
}

/// A picker (files, buffers, symbols, diagnostics, …).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickerState {
    pub prompt: PromptState,
    /// Column names when there is more than one column.
    pub columns: Vec<String>,
    /// The page of matches around the cursor, one cell per visible column.
    pub rows: Vec<Vec<String>>,
    /// The match `rows[0]` is.
    pub offset: u32,
    /// The selected match.
    pub cursor: Option<u32>,
    pub matched: u32,
    pub total: u32,
    /// Matching or item streaming is still going on.
    pub running: bool,
    pub preview: Option<PreviewState>,
}

/// What the picker previews for the selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreviewState {
    pub path: Option<String>,
    /// Lines of the file (or entries of a directory) from `first_line` on.
    pub lines: Vec<String>,
    pub first_line: usize,
    /// The lines the selection points at, zero-based and inclusive.
    pub range: Option<(usize, usize)>,
    /// Shown instead of lines: `<Binary file>` and the like.
    pub placeholder: Option<String>,
}

/// The overlays of one frame.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Overlays {
    pub prompt: Option<PromptState>,
    pub completion: Option<CompletionState>,
    pub info: Option<InfoState>,
    pub popup: Option<PopupState>,
    pub signature: Option<SignatureState>,
    pub picker: Option<PickerState>,
}

/// What the host asks of Helix; handled in Helix's event loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostRequest {
    /// A command as keymaps name it: a static command (`file_picker`) or a
    /// typable one with its arguments (`:write`, `:open a.rs`).
    Command(String),
    /// Input (a key, a paste) for a view: the view is focused first, in
    /// the same queue, so input never lands in the view focused before.
    /// Input for a view that is gone is dropped.
    Input(ViewId, Event),
    /// Host layout: focus a view.
    Focus(ViewId),
    /// Host layout: close a view (its window was closed). The last view
    /// stays: Helix cannot run without one.
    Close(ViewId),
    /// Host layout: a view's size in cells, statusline included.
    Resize(ViewId, u16, u16),
    /// Host layout: a new view for a window of the host, at its size, on
    /// `path` (at `line` and `column`) or a new scratch document. It is reported
    /// as [`HostLayoutEvent::Opened`] with `request`.
    OpenView {
        request: u64,
        path: Option<PathBuf>,
        /// The cursor's line and column, from 1 (column 1 if not given).
        line: Option<usize>,
        column: Option<usize>,
        size: (u16, u16),
    },
    /// Draw everything again (a client attached).
    Redraw,
}

/// A view in host layout, as the host sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewState {
    pub id: ViewId,
    /// Where the view is on the canvas.
    pub area: Rect,
    pub focused: bool,
    /// The document's path; `None` for a scratch document.
    pub path: Option<PathBuf>,
    /// The primary cursor (zero-based line and column in graphemes).
    pub cursor: Position,
    /// One entry per row of the view's text area (the view's height minus
    /// its statusline), top to bottom: the zero-based document line that
    /// starts on that row, as the gutter draws it. `None` for rows where no
    /// line starts: soft-wrap continuations, virtual lines (inline
    /// diagnostics) and rows below the end of the document (the empty line
    /// after a final newline, where the gutter draws `~`, included). Like the
    /// gutter, the top row names its line even when that line started above
    /// it (scrolled into a soft-wrapped line).
    pub rows: Vec<Option<usize>>,
}

/// The views of a host layout and what happened to them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutState {
    pub views: Vec<ViewState>,
    /// The canvas row with Helix's messages and pending keys.
    pub message_row: u16,
    pub events: Vec<HostLayoutEvent>,
}

/// What Helix asks of the host: the typable commands of [`HOST_COMMANDS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostAction {
    /// `:strukta-leader`: the host's leader.
    Leader,
    /// `:strukta-command-line`: the host's command line.
    CommandLine,
}

/// The typable commands that only call the host (registered with `embed`).
pub const HOST_COMMANDS: [&str; 2] = ["strukta-leader", "strukta-command-line"];

/// The host side of the embedding.
pub trait Host: Send + Sync {
    /// The overlays of a frame.
    fn overlays(&self, overlays: Overlays);
    /// A [`HostAction`] Helix asks for.
    fn action(&self, action: HostAction);
    /// The views of a host layout, before each frame.
    fn layout(&self, _layout: LayoutState) {}
}

static HOST_LAYOUT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Before `Application::new`: the host lays out the views (one window per
/// view) instead of Helix's split tree.
pub fn set_host_layout(on: bool) {
    HOST_LAYOUT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Called right after the editor is created, before any view exists.
pub(crate) fn setup_editor(editor: &mut Editor) {
    if HOST_LAYOUT.load(std::sync::atomic::Ordering::Relaxed) {
        editor.tree.host = Some(HostLayout::default());
    }
}

/// Grows or shrinks the backend to the canvas the views need, plus the
/// message row. When the views moved on the canvas, everything is drawn
/// again: the host cuts each view's surface from its place, and a diff
/// against what another view showed there would be wrong.
pub(crate) fn fit_canvas(editor: &mut Editor, compositor: &mut Compositor) {
    if editor.tree.host_places_changed() {
        compositor.need_full_redraw();
    }
    if let Some((width, height)) = editor.tree.host_canvas() {
        let canvas = Rect::new(0, 0, width.max(1), height.saturating_add(1));
        tui::backend::embed::set_size(canvas.width, canvas.height);
        if compositor.size() != canvas {
            compositor.resize(canvas);
        }
    }
}

static HOST: OnceLock<Box<dyn Host>> = OnceLock::new();
static REQUESTS: Mutex<Option<UnboundedReceiver<HostRequest>>> = Mutex::new(None);

/// Installs the host before `Application::new`: from then on the host draws
/// the overlays and Helix handles `requests`. Only the first call counts.
pub fn install(host: Box<dyn Host>, requests: UnboundedReceiver<HostRequest>) {
    if HOST.set(host).is_ok() {
        *REQUESTS.lock().unwrap() = Some(requests);
    }
}

/// Whether a host draws the overlays.
pub fn installed() -> bool {
    HOST.get().is_some()
}

/// The host's requests, for the application's event loop.
pub(crate) fn take_requests() -> Option<UnboundedReceiver<HostRequest>> {
    REQUESTS.lock().unwrap().take()
}

/// The next host request; pending forever without a host.
pub(crate) async fn next_request(
    requests: &mut Option<UnboundedReceiver<HostRequest>>,
) -> Option<HostRequest> {
    match requests {
        Some(requests) => requests.recv().await,
        None => std::future::pending().await,
    }
}

/// Handles a host request: a command runs like a key mapped to it, after
/// the prompts, pickers and popups above the editor are closed. Errors go
/// to the status line. Returns the input for the compositor, if any.
pub(crate) fn handle(
    request: HostRequest,
    editor: &mut Editor,
    compositor: &mut Compositor,
    jobs: &mut Jobs,
) -> Option<Event> {
    let name = match request {
        HostRequest::Command(name) => name,
        HostRequest::Input(view, event) => {
            editor.tree.try_get(view)?;
            editor.focus(view);
            return Some(event);
        }
        HostRequest::Focus(view) => {
            if editor.tree.try_get(view).is_some() {
                editor.focus(view);
            }
            return None;
        }
        HostRequest::Close(view) => {
            if editor.tree.try_get(view).is_some() && editor.tree.views().count() > 1 {
                editor.close(view);
            }
            return None;
        }
        HostRequest::Resize(view, width, height) => {
            if editor.tree.try_get(view).is_some() {
                if let Some(host) = &mut editor.tree.host {
                    host.sizes.insert(view, (width.max(2), height.max(2)));
                }
                editor.tree.recalculate();
                editor.ensure_cursor_in_view(view);
            }
            return None;
        }
        HostRequest::OpenView {
            request,
            path,
            line,
            column,
            size: (width, height),
        } => {
            let size = (width.max(2), height.max(2));
            open_view(editor, request, path, (line, column), size);
            return None;
        }
        HostRequest::Redraw => {
            compositor.need_full_redraw();
            return None;
        }
    };
    let command = match name.parse::<MappableCommand>() {
        Ok(MappableCommand::Macro { .. }) => {
            editor.set_error(format!("No command named '{name}'"));
            return None;
        }
        Ok(command) => command,
        Err(err) => {
            editor.set_error(err.to_string());
            return None;
        }
    };
    let mut cx = compositor::Context {
        editor,
        jobs,
        scroll: None,
    };
    // `Esc` to each layer above the editor.
    while compositor.layers_mut().len() > 1 {
        let layers = compositor.layers_mut().len();
        compositor.handle_event(&Event::Key(crate::key!(Esc)), &mut cx);
        if compositor.layers_mut().len() >= layers {
            break;
        }
    }
    let view = compositor.find::<crate::ui::EditorView>()?;
    if let Some(callback) = view.execute_host_command(&command, &mut cx) {
        callback(compositor, &mut cx);
    }
    None
}

/// [`HostRequest::OpenView`]: a view split off the focused one, on `path`
/// or a scratch document if there is none or it cannot be opened.
fn open_view(
    editor: &mut Editor,
    request: u64,
    path: Option<PathBuf>,
    (line, column): (Option<usize>, Option<usize>),
    size: (u16, u16),
) {
    use helix_view::editor::Action;

    if let Some(host) = &mut editor.tree.host {
        host.opening = Some((request, size));
    }
    let opened = path.is_some_and(|path| match editor.open(&path, Action::VerticalSplit) {
        Ok(_) => true,
        Err(err) => {
            editor.set_error(format!("cannot open {}: {err}", path.display()));
            false
        }
    });
    if opened {
        if let Some(line) = line.filter(|line| *line > 0) {
            goto(editor, line, column.unwrap_or(1));
        }
    } else {
        editor.new_file(Action::VerticalSplit);
    }
    if let Some(host) = &mut editor.tree.host {
        host.opening = None;
    }
}

/// Moves the focused view's cursor to `line` and `column` (from 1, in
/// graphemes, clamped to the text), centered.
fn goto(editor: &mut Editor, line: usize, column: usize) {
    use helix_view::{align_view, Align};

    let (view, doc) = helix_view::current!(editor);
    let text = doc.text().slice(..);
    let line = (line - 1).min(text.len_lines().saturating_sub(1));
    let position = Position::new(line, column.saturating_sub(1));
    let pos = helix_core::pos_at_coords(text, position, true);
    doc.set_selection(view.id, helix_core::Selection::point(pos));
    align_view(doc, view, Align::Center);
}

/// In host layout, the area of the layers above the editor (popups Helix
/// still draws: code actions, `Select`, DAP): the focused view's, so they
/// never spill into another view's window.
pub(crate) fn layer_area(layer: &dyn Component, area: Rect, editor: &Editor) -> Rect {
    match &editor.tree.host {
        Some(_) if !layer.as_any().is::<EditorView>() => editor
            .tree
            .try_get(editor.tree.focus)
            .map_or(area, |view| view.area.intersection(area)),
        _ => area,
    }
}

fn ask_host(action: HostAction, event: PromptEvent) -> anyhow::Result<()> {
    if event == PromptEvent::Validate {
        let host = HOST
            .get()
            .ok_or_else(|| anyhow::anyhow!("no host to ask"))?;
        host.action(action);
    }
    Ok(())
}

/// `:strukta-leader`.
pub(crate) fn leader(
    _: &mut compositor::Context,
    _: Args,
    event: PromptEvent,
) -> anyhow::Result<()> {
    ask_host(HostAction::Leader, event)
}

/// `:strukta-command-line`.
pub(crate) fn command_line(
    _: &mut compositor::Context,
    _: Args,
    event: PromptEvent,
) -> anyhow::Result<()> {
    ask_host(HostAction::CommandLine, event)
}

/// Whether the host draws this compositor layer.
pub fn host_draws(layer: &dyn Component) -> bool {
    let any = layer.as_any();
    installed()
        && (any.is::<Prompt>()
            || any.is::<Popup<Hover>>()
            || any.is::<Popup<Markdown>>()
            || any.is::<Popup<SignatureHelp>>()
            || layer.id() == Some(picker::ID))
}

/// Hands the frame's overlays to the host.
pub(crate) fn report(compositor: &mut Compositor, editor: &mut Editor) {
    let Some(host) = HOST.get() else { return };
    if let Some(layout) = &mut editor.tree.host {
        let events = std::mem::take(&mut layout.events);
        let views: Vec<ViewState> = editor
            .tree
            .views()
            .map(|(view, focused)| ViewState {
                id: view.id,
                area: view.area,
                focused,
                path: editor
                    .document(view.doc)
                    .and_then(|doc| doc.path().map(|path| path.to_path_buf())),
                cursor: editor
                    .document(view.doc)
                    .and_then(|doc| {
                        let text = doc.text().slice(..);
                        let selection = doc.selections().get(&view.id)?;
                        let cursor = selection.primary().cursor(text);
                        Some(helix_core::coords_at_pos(text, cursor))
                    })
                    .unwrap_or_default(),
                rows: view_rows(editor, view),
            })
            .collect();
        let message_row = compositor.size().height.saturating_sub(1);
        host.layout(LayoutState {
            views,
            message_row,
            events,
        });
    }
    let viewport = compositor.size();
    let mut overlays = Overlays {
        info: editor
            .config()
            .auto_info
            .then_some(editor.autoinfo.as_ref())
            .flatten()
            .map(|info| InfoState {
                title: info.title.to_string(),
                text: info.text.clone(),
            }),
        completion: compositor
            .find::<EditorView>()
            .and_then(|view| view.completion.as_mut())
            .map(|completion| completion.embed_state(editor, viewport)),
        ..Overlays::default()
    };
    // Topmost first: the first layer of each kind wins.
    for layer in compositor.layers_mut().iter_mut().rev() {
        if overlays.picker.is_none() && layer.id() == Some(picker::ID) {
            overlays.picker = layer.embed_picker(editor);
            continue;
        }
        let any = layer.as_any_mut();
        if let Some(prompt) = any.downcast_ref::<Prompt>() {
            overlays.prompt.get_or_insert_with(|| prompt.embed_state());
        } else if let Some(popup) = any.downcast_mut::<Popup<SignatureHelp>>() {
            let anchor = anchor(popup, viewport, editor);
            overlays
                .signature
                .get_or_insert_with(|| popup.contents().embed_state(anchor));
        } else if overlays.popup.is_some() {
            continue;
        } else if let Some(popup) = any.downcast_mut::<Popup<Hover>>() {
            overlays.popup = Some(PopupState {
                id: Hover::ID,
                anchor: anchor(popup, viewport, editor),
                markdown: popup.contents().embed_markdown(),
            });
        } else if let Some(popup) = any.downcast_mut::<Popup<Markdown>>() {
            overlays.popup = Some(PopupState {
                id: popup.id().unwrap_or_default(),
                anchor: anchor(popup, viewport, editor),
                markdown: popup.contents().contents().to_string(),
            });
        }
    }
    host.overlays(overlays);
}

/// The document line that starts on each row of a view's text area: the
/// rows `render_document` gives the gutter a line on, computed the same way
/// (soft wrap and virtual lines included) without drawing.
fn view_rows(editor: &Editor, view: &View) -> Vec<Option<usize>> {
    let Some(doc) = editor.document(view.doc) else {
        return Vec::new();
    };
    let inner = view.inner_area(doc);
    let mut rows = vec![None; inner.height as usize];
    let theme = &editor.theme;
    let text = doc.text().slice(..);
    let offset = doc.view_offset(view.id);
    let annotations = view.text_annotations(doc, Some(theme));
    let format = doc.text_format(inner.width, Some(theme));
    let block_row = helix_core::visual_offset_from_block(
        text,
        offset.anchor,
        offset.anchor,
        &format,
        &annotations,
    )
    .0
    .row;
    // The empty line after a final newline is no line: the gutter draws `~`.
    let last_line = text.len_lines() - 1;
    let end = (text.line_to_char(last_line) == text.len_chars()).then_some(last_line);
    let mut last = (usize::MAX, usize::MAX); // (row, line)
    let formatter =
        DocumentFormatter::new_at_prev_checkpoint(text, &format, &annotations, offset.anchor);
    for grapheme in formatter {
        let Some(row) = grapheme.visual_pos.row.checked_sub(block_row) else {
            continue;
        };
        if row >= offset.vertical_offset + rows.len() {
            break;
        }
        if row == last.0 {
            continue;
        }
        let first_visual_line = grapheme.line_idx != last.1;
        last = (row, grapheme.line_idx);
        if first_visual_line && row >= offset.vertical_offset && Some(grapheme.line_idx) != end {
            rows[row - offset.vertical_offset] = Some(grapheme.line_idx);
        }
    }
    rows
}

/// Where a popup points, placed like `Popup::render` does: it stays in place
/// while the cursor moves on its row.
fn anchor<T: Component>(popup: &mut Popup<T>, viewport: Rect, editor: &Editor) -> Position {
    popup.area(viewport, editor);
    popup.get_position().unwrap_or_default()
}
