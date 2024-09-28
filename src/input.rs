use crate::{math as m, Camera};

use winit::dpi::PhysicalPosition;
use winit::event as ev;

pub use ev::{ElementState, MouseButton};
pub use winit::keyboard::KeyCode as Key;

/// This must be at least the number of variants in `Key`
const KEYCODE_COUNT: usize = 200;

/// Tracks the state of input devices so that they can be queried from one place on demand.
#[derive(Clone, Debug)]
pub struct Input {
    // keyboard stored as an array addressed by `Key as usize`.
    // when updating winit, make sure this is as big as the enum!
    keyboard: [AgedState; KEYCODE_COUNT],
    mouse_buttons: MouseButtonState,
    cursor_pos: m::Vec2,
    // previous tick's cursor position to track movements for dragging and such
    prev_cursor_pos: Option<m::Vec2>,
    scroll_delta: f64,
}

impl Input {
    #[inline]
    pub(crate) fn new() -> Self {
        Input {
            keyboard: [AgedState::default(); KEYCODE_COUNT],
            mouse_buttons: Default::default(),
            cursor_pos: m::Vec2::zero(),
            prev_cursor_pos: None,
            scroll_delta: 0.0,
        }
    }

    //
    // Getters
    //

    /// Check the state of a button (keyboard key, mouse button, controller button (TODO))
    /// against a query.
    ///
    /// ```
    /// # use starframe::input::{Input, ButtonQuery, Key};
    /// # let input = Input::new();
    /// if input.button(
    ///     ButtonQuery::kb(Key::Z).held_min(10)
    /// ) {
    ///     // character does a big jump or something idk
    /// }
    /// ```
    #[inline]
    pub fn button(&self, q: ButtonQuery) -> bool {
        let AgedState { state, age } = self.get_button_state(q.button);
        if state != q.state {
            return false;
        }
        if age < q.min_age || age > q.max_age {
            return false;
        }
        true
    }

    /// Get the state of an axis defined by a positive and negative key
    /// or an analog axis (TODO).
    /// Returns a value between -1.0 and 1.0.
    ///
    /// Prefers the positive key if both are pressed.
    #[inline]
    pub fn axis(&self, q: AxisQuery) -> f64 {
        match (
            self.get_button_state(q.pos_btn).state,
            self.get_button_state(q.neg_btn).state,
        ) {
            (ElementState::Pressed, _) => 1.0,
            (_, ElementState::Pressed) => -1.0,
            _ => 0.0,
        }
    }

    /// Get the cursor position in screen space, i.e. origin at the top left,
    /// x right, y down, units of pixels.
    #[inline]
    pub fn cursor_position(&self) -> m::Vec2 {
        self.cursor_pos
    }

    /// Get the cursor movement since last tick in screen space.
    #[inline]
    pub fn cursor_movement(&self) -> m::Vec2 {
        self.prev_cursor_pos
            .map(|prev| self.cursor_pos - prev)
            .unwrap_or_else(m::Vec2::zero)
    }

    /// Get the cursor position in world space, with screen space defined by a camera.
    #[inline]
    pub fn cursor_position_world(&self, camera: &Camera) -> m::Vec2 {
        camera.point_screen_to_world(self.cursor_pos)
    }

    /// Get the cursor movement since last tick in world space,
    /// with screen space defined by a camera.
    #[inline]
    pub fn cursor_movement_world(&self, camera: &Camera) -> m::Vec2 {
        camera.vector_screen_to_world(self.cursor_movement())
    }

    /// Get the vertical scroll distance in pixels during the last tick.
    #[inline]
    pub fn scroll_delta(&self) -> f64 {
        self.scroll_delta
    }

    /// Get the state of a keyboard key along with the number of frames since it last changed.
    #[inline]
    pub fn get_key_state(&self, key: Key) -> AgedState {
        self.keyboard[key as usize]
    }

    #[inline]
    pub fn get_mouse_button_state(&self, mb: MouseButton) -> AgedState {
        *self
            .mouse_buttons
            .get(mb)
            .unwrap_or_else(|| panic!("Untracked mouse button: {:?}", mb))
    }

    #[inline]
    pub fn get_button_state(&self, btn: Button) -> AgedState {
        match btn {
            Button::Keyboard(key) => self.get_key_state(key),
            Button::Mouse(mb) => self.get_mouse_button_state(mb),
        }
    }

    //
    // internal state updates
    //

    /// Do maintenance such as updating the ages of pressed keys.
    /// Call this at the end of every frame.
    ///
    /// Calling is handled internally by [`Game`][crate::game::Game].
    #[inline]
    pub(crate) fn tick(&mut self) {
        for state in &mut self.keyboard {
            state.age += 1;
        }

        self.mouse_buttons.left.age += 1;
        self.mouse_buttons.middle.age += 1;
        self.mouse_buttons.right.age += 1;

        self.scroll_delta = 0.0;
        self.prev_cursor_pos = Some(self.cursor_pos);
    }

    /// Track the effect of a keyboard event.
    #[inline]
    pub(crate) fn track_keyboard(&mut self, evt: &ev::KeyEvent) {
        if let winit::keyboard::PhysicalKey::Code(code) = evt.physical_key {
            let cached_key = &mut self.keyboard[code as usize];
            if evt.state != cached_key.state {
                *cached_key = AgedState::new(evt.state);
            }
        }
    }

    /// Perform whatever tracking is available for the given window event.
    #[inline]
    pub(crate) fn track_window_event(&mut self, event: &ev::WindowEvent) {
        use ev::WindowEvent::*;
        match event {
            KeyboardInput { event, .. } => self.track_keyboard(event),
            MouseInput { button, state, .. } => self.track_mouse_button(*button, *state),
            MouseWheel { delta, .. } => self.track_mouse_wheel(*delta),
            CursorMoved { position, .. } => self.track_cursor_movement(*position),
            _ => (),
        }
    }

    /// Track a mouse button event.
    #[inline]
    fn track_mouse_button(&mut self, button: ev::MouseButton, new_state: ElementState) {
        if let Some(s) = self.mouse_buttons.get_mut(button) {
            *s = AgedState::new(new_state);
        }
    }

    #[inline]
    fn track_cursor_movement(&mut self, pos: PhysicalPosition<f64>) {
        self.cursor_pos = m::Vec2::new(pos.x as f32, pos.y as f32);
    }

    /// Track a mouse wheel movement.
    ///
    /// TODO: test to make line and pixel delta effects match
    #[inline]
    fn track_mouse_wheel(&mut self, delta: ev::MouseScrollDelta) {
        const PIXELS_PER_LINE: f64 = 10.0;

        use ev::MouseScrollDelta::*;
        match delta {
            LineDelta(_, y) => self.scroll_delta += PIXELS_PER_LINE * y as f64,
            PixelDelta(PhysicalPosition { y, .. }) => self.scroll_delta += y,
        }
    }
}

//
// queries
//

/// A button on any input device (keyboard, mouse, TODO: gamepad).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde-types", derive(serde::Deserialize, serde::Serialize))]
pub enum Button {
    Keyboard(Key),
    Mouse(MouseButton),
}
impl From<Key> for Button {
    fn from(k: Key) -> Self {
        Button::Keyboard(k)
    }
}
impl From<MouseButton> for Button {
    fn from(mb: MouseButton) -> Self {
        Button::Mouse(mb)
    }
}

/// A query for matching against the state of a button.
/// See [InputCache::button][`self::InputCache::button] for usage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde-types", derive(serde::Deserialize, serde::Serialize))]
pub struct ButtonQuery {
    pub button: Button,
    pub state: ElementState,
    pub min_age: usize,
    pub max_age: usize,
}

impl ButtonQuery {
    /// Query a button on either the mouse or the keyboard.
    ///
    /// With no additional modifiers the query checks for that key
    /// having just been pressed down after last frame.
    #[inline]
    pub fn new(btn: Button) -> Self {
        match btn {
            Button::Keyboard(key) => Self::kb(key),
            Button::Mouse(mb) => Self::mouse(mb),
        }
    }

    /// Query a keyboard key.
    ///
    /// With no additional modifiers the query checks for that key
    /// having just been pressed down after last frame.
    ///
    /// `From<Key>` is also implemented for `ButtonQuery`,
    /// so these two are equivalent:
    /// ```
    /// # use starframe::input::{ButtonQuery, Key};
    /// assert_eq!(
    ///     ButtonQuery::kb(Key::P),
    ///     Key::P.into(),
    /// );
    /// ```
    #[inline]
    pub fn kb(key: Key) -> Self {
        Self {
            button: Button::Keyboard(key),
            state: ElementState::Pressed,
            min_age: 0,
            max_age: 0,
        }
    }

    /// Query a mouse button.
    ///
    /// With no additional modifiers the query checks for that button
    /// having just been pressed down after last frame.
    #[inline]
    pub fn mouse(btn: MouseButton) -> Self {
        Self {
            button: Button::Mouse(btn),
            state: ElementState::Pressed,
            min_age: 0,
            max_age: 0,
        }
    }

    /// Modify the query to look for the released state instead of pressed.
    ///
    /// Can be combined with the different variants of `held_*` to match keys
    /// that have been released for some amount of time.
    /// ```
    /// # use starframe::input::{ButtonQuery, Key};
    /// let havent_pressed_x_for_a_while = ButtonQuery::kb(Key::X)
    ///     .released()
    ///     .held_exact(42);
    /// ```
    #[inline]
    pub fn released(mut self) -> Self {
        self.state = ElementState::Released;
        self
    }

    /// Match if the button has been held for any amount of time.
    ///
    /// This overwrites the parameters set by any other variant of `held_*`.
    #[inline]
    pub fn held(mut self) -> Self {
        self.min_age = 0;
        self.max_age = usize::MAX;
        self
    }

    /// Match if the button has been held for exactly the given number of frames.
    ///
    /// This overwrites the parameters set by any other variant of `held_*`.
    #[inline]
    pub fn held_exact(mut self, frames: usize) -> Self {
        self.min_age = frames;
        self.max_age = frames;
        self
    }

    /// Match if the button has been held for at least the given number of frames.
    ///
    /// This overwrites the parameters set by any other variant of `held_*`.
    #[inline]
    pub fn held_min(mut self, frames: usize) -> Self {
        self.min_age = frames;
        self.max_age = usize::MAX;
        self
    }

    /// Match if the button has been held for at most the given number of frames.
    ///
    /// This overwrites the parameters set by any other variant of `held_*`.
    #[inline]
    pub fn held_max(mut self, frames: usize) -> Self {
        self.min_age = 0;
        self.max_age = frames;
        self
    }

    /// Match if the button has been held for a number of frames between the given values.
    ///
    /// This overwrites the parameters set by any other variant of `held_*`.
    #[inline]
    pub fn held_range(mut self, min_frames: usize, max_frames: usize) -> Self {
        self.min_age = min_frames;
        self.max_age = max_frames;
        self
    }
}

impl From<Button> for ButtonQuery {
    fn from(btn: Button) -> Self {
        Self::new(btn)
    }
}

impl From<Key> for ButtonQuery {
    fn from(k: Key) -> Self {
        Self::kb(k)
    }
}

impl From<MouseButton> for ButtonQuery {
    fn from(mb: MouseButton) -> Self {
        Self::mouse(mb)
    }
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde-types", derive(serde::Deserialize, serde::Serialize))]
pub struct AxisQuery {
    pub pos_btn: Button,
    pub neg_btn: Button,
}

//
// state types
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

// Mouse

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
            // TODO: track back and forward buttons too
            MB::Forward => None,
            MB::Back => None,
            MB::Other(_) => None,
        }
    }

    pub fn get_mut(&mut self, button: MouseButton) -> Option<&mut AgedState> {
        use MouseButton as MB;
        match button {
            MB::Left => Some(&mut self.left),
            MB::Middle => Some(&mut self.middle),
            MB::Right => Some(&mut self.right),
            MB::Forward => None,
            MB::Back => None,
            MB::Other(_) => None,
        }
    }
}
