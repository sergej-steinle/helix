//! A backend for embedding Helix in another application: frames are handed
//! to a sink instead of being written to a terminal. The host sets the size
//! and receives every frame as the cells that changed since the last one.

use crate::{backend::Backend, buffer::Cell, terminal::Config};
use helix_view::graphics::{CursorKind, Rect};
use helix_view::theme::{Color, Mode};
use std::io;
use std::sync::{Arc, Mutex, OnceLock};

/// One frame: the cells that changed, where the cursor is, and whether the
/// host has to clear its grid first.
#[derive(Debug, Clone, Default)]
pub struct EmbedFrame {
    pub width: u16,
    pub height: u16,
    pub clear: bool,
    pub cells: Vec<(u16, u16, Cell)>,
    /// `None` while the cursor is hidden.
    pub cursor: Option<(u16, u16, CursorKind)>,
    pub background: Option<Color>,
}

/// Receives the frames.
pub type EmbedSink = Box<dyn FnMut(EmbedFrame) + Send>;

/// What the host shares with the backend.
pub struct EmbedHost {
    pub sink: Mutex<EmbedSink>,
    /// `(width, height)` in cells.
    pub size: Mutex<(u16, u16)>,
    pub mode: Option<Mode>,
}

static HOST: OnceLock<Arc<EmbedHost>> = OnceLock::new();

/// Installs the host before the application creates its backend. Only the
/// first call counts.
pub fn install(host: Arc<EmbedHost>) {
    let _ = HOST.set(host);
}

pub struct EmbedBackend {
    host: Arc<EmbedHost>,
    frame: EmbedFrame,
    cursor_visible: bool,
    cursor: (u16, u16, CursorKind),
}

impl EmbedBackend {
    /// The backend of the installed host; panics without one.
    pub fn new() -> Self {
        let host = Arc::clone(HOST.get().expect("embed host not installed"));
        Self {
            host,
            frame: EmbedFrame::default(),
            cursor_visible: false,
            cursor: (0, 0, CursorKind::Block),
        }
    }

    fn emit(&mut self) {
        let (width, height) = *self.host.size.lock().unwrap();
        let mut frame = std::mem::take(&mut self.frame);
        frame.width = width;
        frame.height = height;
        frame.cursor = self.cursor_visible.then_some(self.cursor);
        (self.host.sink.lock().unwrap())(frame);
    }
}

impl Default for EmbedBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for EmbedBackend {
    fn claim(&mut self) -> Result<(), io::Error> {
        Ok(())
    }

    fn reconfigure(&mut self, _config: Config) -> Result<(), io::Error> {
        Ok(())
    }

    fn restore(&mut self) -> Result<(), io::Error> {
        Ok(())
    }

    fn draw<'a, I>(&mut self, content: I) -> Result<(), io::Error>
    where
        I: Iterator<Item = (u16, u16, &'a Cell)>,
    {
        self.frame
            .cells
            .extend(content.map(|(x, y, cell)| (x, y, cell.clone())));
        Ok(())
    }

    fn hide_cursor(&mut self) -> Result<(), io::Error> {
        self.cursor_visible = false;
        Ok(())
    }

    fn show_cursor(&mut self, kind: CursorKind) -> Result<(), io::Error> {
        self.cursor_visible = true;
        self.cursor.2 = kind;
        Ok(())
    }

    fn set_cursor(&mut self, x: u16, y: u16) -> Result<(), io::Error> {
        self.cursor.0 = x;
        self.cursor.1 = y;
        Ok(())
    }

    fn clear(&mut self) -> Result<(), io::Error> {
        self.frame.cells.clear();
        self.frame.clear = true;
        Ok(())
    }

    fn start_sync(&mut self) -> Result<(), io::Error> {
        Ok(())
    }

    fn end_sync(&mut self) -> Result<(), io::Error> {
        Ok(())
    }

    fn size(&self) -> Result<Rect, io::Error> {
        let (width, height) = *self.host.size.lock().unwrap();
        Ok(Rect::new(0, 0, width, height))
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        self.emit();
        Ok(())
    }

    fn supports_true_color(&self) -> bool {
        true
    }

    fn get_theme_mode(&self) -> Option<Mode> {
        self.host.mode
    }

    fn set_background_color(&mut self, color: Option<Color>) -> io::Result<()> {
        self.frame.background = color;
        Ok(())
    }
}
