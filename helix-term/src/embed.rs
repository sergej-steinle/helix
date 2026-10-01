//! Embedding Helix in a host application (feature `embed`): after every
//! frame the host gets the overlays it draws itself, as data. Overlays the
//! host draws are left out of the cell frames; Helix keeps handling their
//! keys.

use std::sync::OnceLock;

use helix_core::Position;
use helix_view::{graphics::Rect, Editor};

use crate::compositor::{Component, Compositor};
use crate::ui::lsp::{hover::Hover, signature_help::SignatureHelp};
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

type Hook = Box<dyn Fn(Overlays) + Send + Sync>;

static HOOK: OnceLock<Hook> = OnceLock::new();

/// Installs the host's hook; from then on the host draws the overlays. Only
/// the first call counts.
pub fn install(hook: Hook) {
    let _ = HOOK.set(hook);
}

/// Whether a host draws the overlays.
pub fn installed() -> bool {
    HOOK.get().is_some()
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
    let Some(hook) = HOOK.get() else { return };
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
    hook(overlays);
}

/// Where a popup points, placed like `Popup::render` does: it stays in place
/// while the cursor moves on its row.
fn anchor<T: Component>(popup: &mut Popup<T>, viewport: Rect, editor: &Editor) -> Position {
    popup.area(viewport, editor);
    popup.get_position().unwrap_or_default()
}
