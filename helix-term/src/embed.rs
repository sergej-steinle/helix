//! Embedding Helix in a host application (feature `embed`): after every
//! frame the host gets the overlays it draws itself, as data. Overlays the
//! host draws are left out of the cell frames.

use std::sync::OnceLock;

use crate::compositor::{Component, Compositor};
use crate::ui::Prompt;

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

/// The overlays of one frame.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Overlays {
    pub prompt: Option<PromptState>,
}

type Hook = Box<dyn Fn(Overlays) + Send + Sync>;

static HOOK: OnceLock<Hook> = OnceLock::new();

/// Installs the host's hook; from then on the host draws prompts. Only the
/// first call counts.
pub fn install(hook: Hook) {
    let _ = HOOK.set(hook);
}

/// Whether the host draws prompt layers.
pub fn host_draws(layer: &dyn Component) -> bool {
    HOOK.get().is_some() && layer.as_any().is::<Prompt>()
}

/// Hands the frame's overlays to the host.
pub(crate) fn report(compositor: &Compositor) {
    let Some(hook) = HOOK.get() else { return };
    let prompt = compositor
        .layers()
        .iter()
        .rev()
        .find_map(|layer| layer.as_any().downcast_ref::<Prompt>())
        .map(Prompt::embed_state);
    hook(Overlays { prompt });
}
