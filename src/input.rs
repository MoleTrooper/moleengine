use crate::math as m;

use winit::dpi::PhysicalPosition;

use winit::event as ev;

pub use ev::ElementState;
pub use ev::MouseButton;
pub use ev::VirtualKeyCode as Key;

/// Track the state of input devices so that they can be looked up from a single location
/// instead of moving window events around.
#[derive(Clone, Debug)]
pub struct InputCache {
    // keyboard stored as an array addressed by `Key as usize`.
    // when updating winit, make sure this is as big as the enum!
    keyboard: [AgedState; 163],
    mouse_buttons: MouseButtonState,
    cursor_pos: CursorPosition,
    scroll_delta: f64,
    drag_state: Option<DragState>,
}

impl InputCache {
    pub fn new() -> Self {
        InputCache {
            keyboard: [AgedState::default(); 163],
            mouse_buttons: Default::default(),
            cursor_pos: CursorPosition::OutOfWindow(PhysicalPosition::new(0.0, 0.0)),
            scroll_delta: 0.0,
            drag_state: None,
        }
    }

    /// Do maintenance such as updating the ages of pressed keys.
    /// Call this at the end of every frame.
    ///
    /// Calling is handled internally by [`Game`][crate::game::Game].
    pub(crate) fn tick(&mut self) {
        for state in &mut self.keyboard {
            state.age += 1;
        }

        self.mouse_buttons.left.age += 1;
        self.mouse_buttons.middle.age += 1;
        self.mouse_buttons.right.age += 1;

        self.scroll_delta = 0.0;

        match self.drag_state {
            Some(DragState::InProgress {
                ref mut duration, ..
            }) => *duration += 1,
            Some(DragState::Completed { .. }) => self.drag_state = None,
            None => (),
        }
    }

    //
    // Getters
    //

    /// Get the state of a keyboard key along with the number of frames since it last changed.
    pub fn get_key_state(&self, key: Key) -> AgedState {
        self.keyboard[key as usize]
    }

    /// True if the requested key is currently pressed
    /// (for fewer frames than age_limit if provided), false otherwise.
    #[inline]
    pub fn is_key_pressed(&self, key: Key, age_limit: Option<usize>) -> bool {
        self.is_key_in_state(key, ElementState::Pressed, age_limit)
    }

    /// True if the requested key is currently released
    /// (for fewer frames than age_limit if provided), false otherwise.
    #[inline]
    pub fn is_key_released(&self, key: Key, age_limit: Option<usize>) -> bool {
        self.is_key_in_state(key, ElementState::Released, age_limit)
    }

    fn is_key_in_state(
        &self,
        key: Key,
        wanted_state: ElementState,
        age_limit: Option<usize>,
    ) -> bool {
        let AgedState { state, age } = self.get_key_state(key);
        if state == wanted_state {
            if let Some(al) = age_limit {
                age <= al
            } else {
                true
            }
        } else {
            false
        }
    }

    /// Get the state of an axis defined by a positive and negavite key.
    /// Prefers the positive key if both are pressed.
    pub fn get_key_axis_state(&self, pos_key: Key, neg_key: Key) -> KeyAxisState {
        match (
            self.get_key_state(pos_key).state,
            self.get_key_state(neg_key).state,
        ) {
            (ElementState::Pressed, _) => KeyAxisState::Pos,
            (_, ElementState::Pressed) => KeyAxisState::Neg,
            _ => KeyAxisState::Zero,
        }
    }

    /// True if the requested mouse button is currently pressed
    /// (for fewer frames than age_limit if provided), false otherwise.
    /// # Panics
    /// Panics if the requested mouse button is not tracked.
    /// Left, Middle and Right are tracked by default.
    pub fn is_mouse_button_pressed(
        &self,
        button: ev::MouseButton,
        age_limit: Option<usize>,
    ) -> bool {
        let AgedState { age, state } = self
            .mouse_buttons
            .get(button)
            .unwrap_or_else(|| panic!("Untracked mouse button: {:?}", button));

        if let ElementState::Pressed = state {
            if let Some(al) = age_limit {
                *age <= al
            } else {
                true
            }
        } else {
            false
        }
    }

    /// Get the cursor position in logical pixels down and right from the top left.
    pub fn cursor_position(&self) -> &CursorPosition {
        &self.cursor_pos
    }

    /// Get the vertical scroll distance in pixels during the last tick.
    pub fn scroll_delta(&self) -> f64 {
        self.scroll_delta
    }

    pub fn drag_state(&self) -> &Option<DragState> {
        &self.drag_state
    }

    //
    // Trackers
    //

    /// Track the effect of a keyboard event.
    pub fn track_keyboard(&mut self, evt: ev::KeyboardInput) {
        if let Some(code) = evt.virtual_keycode {
            let cached_key = &mut self.keyboard[code as usize];
            if evt.state != cached_key.state {
                *cached_key = AgedState::new(evt.state);
            }
        }
    }

    /// Perform whatever tracking is available for the given window event.
    pub fn track_window_event(&mut self, event: &ev::WindowEvent) {
        use ev::WindowEvent::*;
        match event {
            KeyboardInput { input, .. } => self.track_keyboard(*input),
            MouseInput { button, state, .. } => self.track_mouse_button(*button, *state),
            MouseWheel { delta, .. } => self.track_mouse_wheel(*delta),
            CursorMoved { position, .. } => self.track_cursor_movement(*position),
            CursorEntered { .. } => self.track_cursor_enter(),
            CursorLeft { .. } => self.track_cursor_leave(),
            _ => (),
        }
    }

    /// Track a mouse button event.
    pub fn track_mouse_button(&mut self, button: ev::MouseButton, new_state: ElementState) {
        if let Some(s) = self.mouse_buttons.get_mut(button) {
            *s = AgedState::new(new_state);
        }

        // drag, at least for now hardcoded to only work with left click
        match (button, new_state, self.drag_state) {
            (ev::MouseButton::Left, ElementState::Pressed, None) => self.begin_drag(),
            (ev::MouseButton::Left, ElementState::Released, _) => self.finish_drag(),
            _ => (),
        }
    }

    /// Track the screen position of the mouse cursor.
    pub fn track_cursor_movement(&mut self, position: PhysicalPosition<f64>) {
        *self.cursor_pos.get_mut() = position;
    }

    pub fn track_cursor_enter(&mut self) {
        self.cursor_pos = CursorPosition::InWindow(self.cursor_pos.take());
    }

    pub fn track_cursor_leave(&mut self) {
        self.cursor_pos = CursorPosition::OutOfWindow(self.cursor_pos.take());
    }

    /// Track a mouse wheel movement.
    ///
    /// TODO: test to make line and pixel delta effects match
    ///
    pub fn track_mouse_wheel(&mut self, delta: ev::MouseScrollDelta) {
        const PIXELS_PER_LINE: f64 = 10.0;

        use ev::MouseScrollDelta::*;
        match delta {
            LineDelta(_, y) => self.scroll_delta += PIXELS_PER_LINE * y as f64,
            PixelDelta(PhysicalPosition { y, .. }) => self.scroll_delta += y as f64,
        }
    }

    fn begin_drag(&mut self) {
        self.drag_state = Some(DragState::InProgress {
            start: *self.cursor_pos.get(),
            duration: 0,
        });
    }

    fn finish_drag(&mut self) {
        if let Some(DragState::InProgress { start, duration }) = self.drag_state {
            self.drag_state = Some(DragState::Completed {
                start,
                duration,
                end: *self.cursor_pos.get(),
            });
        }
    }
}

impl Default for InputCache {
    fn default() -> Self {
        Self::new()
    }
}

//

/// The state of a button (keyboard key or mouse button)
/// and time in number of ticks since last state change.
#[derive(Clone, Copy, Debug)]
pub struct AgedState {
    pub state: ElementState,
    pub age: usize,
}

impl AgedState {
    pub fn new(state: ElementState) -> Self {
        AgedState { state, age: 0 }
    }
}

impl Default for AgedState {
    fn default() -> Self {
        Self::new(ElementState::Released)
    }
}

/// The state of an input axis defined by a positive and negative key.
pub enum KeyAxisState {
    Pos,
    Zero,
    Neg,
}

// Mouse

/// Cursor position taking into account whether it's in the window or not.
/// Usually you don't want to do anything if you're outside the window.
#[derive(Clone, Copy, Debug)]
pub enum CursorPosition {
    InWindow(PhysicalPosition<f64>),
    OutOfWindow(PhysicalPosition<f64>),
}

impl CursorPosition {
    pub fn get(&self) -> &PhysicalPosition<f64> {
        match self {
            CursorPosition::InWindow(p) => p,
            CursorPosition::OutOfWindow(p) => p,
        }
    }

    pub fn get_mut(&mut self) -> &mut PhysicalPosition<f64> {
        match self {
            CursorPosition::InWindow(p) => p,
            CursorPosition::OutOfWindow(p) => p,
        }
    }

    pub fn take(self) -> PhysicalPosition<f64> {
        match self {
            CursorPosition::InWindow(p) => p,
            CursorPosition::OutOfWindow(p) => p,
        }
    }
}

impl From<&CursorPosition> for m::Vec2 {
    fn from(cp: &CursorPosition) -> m::Vec2 {
        let pos = cp.get();
        m::Vec2::new(pos.x, pos.y)
    }
}

//

#[derive(Clone, Copy, Debug, Default)]
struct MouseButtonState {
    left: AgedState,
    middle: AgedState,
    right: AgedState,
}

impl MouseButtonState {
    pub fn get(&self, button: MouseButton) -> Option<&AgedState> {
        use MouseButton as MB;
        match button {
            MB::Left => Some(&self.left),
            MB::Middle => Some(&self.middle),
            MB::Right => Some(&self.right),
            MB::Other(_) => None,
        }
    }

    pub fn get_mut(&mut self, button: MouseButton) -> Option<&mut AgedState> {
        use MouseButton as MB;
        match button {
            MB::Left => Some(&mut self.left),
            MB::Middle => Some(&mut self.middle),
            MB::Right => Some(&mut self.right),
            MB::Other(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DragState {
    InProgress {
        start: PhysicalPosition<f64>,
        duration: u32,
    },
    Completed {
        start: PhysicalPosition<f64>,
        end: PhysicalPosition<f64>,
        duration: u32,
    },
}
