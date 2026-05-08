//! Grapple / constraint plugin for Quartz.
//!
//! Provides rope, spring, and grapple/swinging mechanics.
//! Fully self-contained — stores all state internally.
//!
//! Dispatch grapple commands via `Action::PluginCall` with a `GrappleCommand` payload.
//! The plugin manages `grapple_constraints` internally and enforces them
//! in `on_post_solve` after the physics step.
//!
//! `GrapplePlugin` must be registered with `canvas.add_plugin(GrapplePlugin::new())`
//! before any grapple commands can be dispatched.

use crate::{Canvas, plugin::QuartzPlugin, types::{Target, Location}};
use std::collections::HashMap;

// ── Solve helper ──────────────────────────────────────────────────────────────

/// Solve a distance constraint between two points.
/// Returns the corrective impulse to apply to the dynamic point (pos_a).
pub fn solve_distance_constraint(
    pos_a:       (f32, f32),
    pos_b:       (f32, f32),
    rest_length: f32,
    stiffness:   f32,
    damping:     f32,
    vel_a:       (f32, f32),
) -> (f32, f32) {
    let dx   = pos_a.0 - pos_b.0;
    let dy   = pos_a.1 - pos_b.1;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < 0.001 { return (0.0, 0.0); }
    let nx          = dx / dist;
    let ny          = dy / dist;
    let stretch     = dist - rest_length;
    let spring_f    = -stiffness * stretch;
    let radial_vel  = vel_a.0 * nx + vel_a.1 * ny;
    let damp_f      = -damping * radial_vel;
    let total       = spring_f + damp_f;
    (nx * total, ny * total)
}

// ── DistanceConstraint ────────────────────────────────────────────────────────

/// A simple distance constraint between two points (rope, tether, rigid link).
#[derive(Clone, Debug)]
pub struct DistanceConstraint {
    pub anchor:      (f32, f32),
    pub rest_length: f32,
    pub stiffness:   f32,
    pub damping:     f32,
    pub active:      bool,
}

impl Default for DistanceConstraint {
    fn default() -> Self {
        Self { anchor: (0.0, 0.0), rest_length: 100.0, stiffness: 0.5, damping: 0.1, active: true }
    }
}

impl DistanceConstraint {
    pub fn new(anchor: (f32, f32), rest_length: f32) -> Self {
        Self { anchor, rest_length, ..Default::default() }
    }
    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness.clamp(0.0, 1.0); self
    }
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.clamp(0.0, 1.0); self
    }
    pub fn solve(&self, pos: (f32, f32), vel: (f32, f32)) -> (f32, f32) {
        if !self.active { return (0.0, 0.0); }
        solve_distance_constraint(pos, self.anchor, self.rest_length, self.stiffness, self.damping, vel)
    }
}

// ── SpringConstraint ──────────────────────────────────────────────────────────

/// A spring connecting two points. Softer than distance constraint — suitable
/// for bouncy tethers, bungee cords, elastic connections.
#[derive(Clone, Debug)]
pub struct SpringConstraint {
    pub anchor:      (f32, f32),
    pub rest_length: f32,
    pub spring_k:    f32,
    pub damp_k:      f32,
    pub active:      bool,
}

impl Default for SpringConstraint {
    fn default() -> Self {
        Self { anchor: (0.0, 0.0), rest_length: 100.0, spring_k: 200.0, damp_k: 10.0, active: true }
    }
}

impl SpringConstraint {
    pub fn new(anchor: (f32, f32), rest_length: f32) -> Self {
        Self { anchor, rest_length, ..Default::default() }
    }
    pub fn with_spring_k(mut self, k: f32) -> Self { self.spring_k = k.max(0.0); self }
    pub fn with_damp_k(mut self, k: f32) -> Self   { self.damp_k = k.max(0.0); self }

    pub fn solve(&self, pos: (f32, f32), vel: (f32, f32)) -> (f32, f32) {
        if !self.active { return (0.0, 0.0); }
        let dx   = pos.0 - self.anchor.0;
        let dy   = pos.1 - self.anchor.1;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 0.001 { return (0.0, 0.0); }
        let nx         = dx / dist;
        let ny         = dy / dist;
        let stretch    = dist - self.rest_length;
        let spring_f   = -self.spring_k * stretch;
        let radial_vel = vel.0 * nx + vel.1 * ny;
        let damp_f     = -self.damp_k * radial_vel;
        let total      = spring_f + damp_f;
        (nx * total, ny * total)
    }
}

// ── SwingBias ─────────────────────────────────────────────────────────────────

/// Directional bias for a grapple's swing motion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SwingBias {
    None,
    Horizontal,
    Vertical,
}

impl Default for SwingBias {
    fn default() -> Self { SwingBias::None }
}

// ── GrappleCorrection ─────────────────────────────────────────────────────────

/// Result of solving a grapple constraint for one frame.
#[derive(Clone, Debug)]
pub struct GrappleCorrection {
    pub position: Option<(f32, f32)>,
    pub velocity: Option<(f32, f32)>,
}

impl GrappleCorrection {
    pub fn none() -> Self { Self { position: None, velocity: None } }
    pub fn applied(&self) -> bool { self.position.is_some() }
}

// ── GrappleConstraint ─────────────────────────────────────────────────────────

/// A grapple/swinging constraint. Designed for pendulum-style swinging mechanics.
///
/// Uses XPBD position-level correction — rope feels rigid by default.
/// Stiffness 1.0 = fully rigid, 0.0 = no enforcement.
#[derive(Clone, Debug)]
pub struct GrappleConstraint {
    pub anchor:         (f32, f32),
    pub anchor_object:  Option<String>,
    pub length:         f32,
    pub stiffness:      f32,
    pub damping:        f32,
    pub max_swing_speed: f32,
    pub auto_shorten:   bool,
    pub swing_bias:     SwingBias,
    pub active:         bool,
}

impl Default for GrappleConstraint {
    fn default() -> Self {
        Self {
            anchor:          (0.0, 0.0),
            anchor_object:   None,
            length:          200.0,
            stiffness:       0.8,
            damping:         0.05,
            max_swing_speed: 0.0,
            auto_shorten:    false,
            swing_bias:      SwingBias::None,
            active:          true,
        }
    }
}

impl GrappleConstraint {
    pub fn at_point(anchor: (f32, f32), length: f32) -> Self {
        Self { anchor, length: length.max(1.0), ..Default::default() }
    }

    pub fn to_object(object_name: impl Into<String>, length: f32) -> Self {
        Self { anchor_object: Some(object_name.into()), length: length.max(1.0), ..Default::default() }
    }

    pub fn with_stiffness(mut self, stiffness: f32) -> Self {
        self.stiffness = stiffness.clamp(0.0, 1.0); self
    }
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.clamp(0.0, 1.0); self
    }
    pub fn with_max_swing_speed(mut self, speed: f32) -> Self {
        self.max_swing_speed = speed.max(0.0); self
    }
    pub fn with_auto_shorten(mut self) -> Self {
        self.auto_shorten = true; self
    }
    pub fn with_swing_bias(mut self, bias: SwingBias) -> Self {
        self.swing_bias = bias; self
    }

    /// Compute the grapple correction for this frame (XPBD position-level).
    pub fn solve(&mut self, obj_pos: (f32, f32), obj_vel: (f32, f32)) -> GrappleCorrection {
        if !self.active { return GrappleCorrection::none(); }

        let dx   = obj_pos.0 - self.anchor.0;
        let dy   = obj_pos.1 - self.anchor.1;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 0.001 { return GrappleCorrection::none(); }

        if self.auto_shorten && dist < self.length { self.length = dist; }

        if dist <= self.length { return GrappleCorrection::none(); }

        let nx = dx / dist;
        let ny = dy / dist;

        let corrected_x = self.anchor.0 + nx * self.length;
        let corrected_y = self.anchor.1 + ny * self.length;
        let cx = obj_pos.0 + (corrected_x - obj_pos.0) * self.stiffness;
        let cy = obj_pos.1 + (corrected_y - obj_pos.1) * self.stiffness;

        let (tangent_x, tangent_y) = (-ny, nx);
        let radial_vel  = obj_vel.0 * nx + obj_vel.1 * ny;
        let tangent_vel = obj_vel.0 * tangent_x + obj_vel.1 * tangent_y;

        let corrected_radial = if radial_vel > 0.0 {
            radial_vel * (1.0 - self.stiffness)
        } else {
            radial_vel
        };
        let damped_tangent = tangent_vel * (1.0 - self.damping);

        let mut new_vx = nx * corrected_radial + tangent_x * damped_tangent;
        let mut new_vy = ny * corrected_radial + tangent_y * damped_tangent;

        match self.swing_bias {
            SwingBias::None       => {}
            SwingBias::Horizontal => { new_vy = obj_vel.1 + (new_vy - obj_vel.1) * 0.6; }
            SwingBias::Vertical   => { new_vx = obj_vel.0 + (new_vx - obj_vel.0) * 0.6; }
        }

        if self.max_swing_speed > 0.0 {
            let speed = (new_vx * new_vx + new_vy * new_vy).sqrt();
            if speed > self.max_swing_speed {
                let scale = self.max_swing_speed / speed;
                new_vx *= scale;
                new_vy *= scale;
            }
        }

        GrappleCorrection {
            position: Some((cx, cy)),
            velocity: Some((new_vx, new_vy)),
        }
    }
}

// ── Presets ───────────────────────────────────────────────────────────────────

impl GrappleConstraint {
    pub fn grappling_hook(anchor: (f32, f32), length: f32) -> Self {
        Self::at_point(anchor, length).with_stiffness(0.9).with_damping(0.05)
    }
    pub fn web_swing(anchor: (f32, f32), length: f32) -> Self {
        Self::at_point(anchor, length)
            .with_stiffness(0.7).with_damping(0.02)
            .with_swing_bias(SwingBias::Horizontal)
    }
    pub fn bungee(anchor: (f32, f32), length: f32) -> Self {
        Self::at_point(anchor, length).with_stiffness(0.3).with_damping(0.15)
    }
    pub fn rigid_tether(anchor: (f32, f32), length: f32) -> Self {
        Self::at_point(anchor, length).with_stiffness(1.0).with_damping(0.1)
    }
    pub fn wrecking_ball(anchor: (f32, f32), length: f32) -> Self {
        Self::at_point(anchor, length)
            .with_stiffness(0.95).with_damping(0.02)
            .with_max_swing_speed(800.0)
    }
}

// ── GrappleCommand ────────────────────────────────────────────────────────

/// Commands dispatched to GrapplePlugin via Action::PluginCall.
#[derive(Clone, Debug)]
pub enum GrappleCommand {
    /// Attach a grapple constraint to a target object.
    Attach { target: Target, grapple: GrappleConstraint },
    /// Release (remove) the grapple from a target object.
    Release { target: Target },
    /// Set the rope length of an active grapple.
    SetLength { target: Target, value: f32 },
    /// Set the stiffness of an active grapple (0.0–1.0).
    SetStiffness { target: Target, value: f32 },
    /// Set the damping of an active grapple (0.0–1.0).
    SetDamping { target: Target, value: f32 },
    /// Move the anchor of an active grapple to a new world position.
    SetAnchor { target: Target, x: f32, y: f32 },
    /// Re-target the grapple anchor to follow a named object.
    SetAnchorObject { target: Target, anchor_object: String },
    /// Set the swing bias of an active grapple.
    SetSwingBias { target: Target, bias: SwingBias },
}

// ── GrapplePlugin ─────────────────────────────────────────────────────────

/// Plugin that manages grapple constraints for the engine.
///
/// Stores all grapple state internally and enforces constraints during
/// the `on_post_solve` hook (after the physics step).
///
/// Dispatch commands via `canvas.run(Action::PluginCall { name: "grapple".into(), payload: Arc::new(...) })`.
pub struct GrapplePlugin {
    /// Per-object grapple constraints. Key = game object name.
    pub(crate) grapple_constraints: HashMap<String, GrappleConstraint>,
}

impl GrapplePlugin {
    pub fn new() -> Self {
        Self { grapple_constraints: HashMap::new() }
    }
}

impl Default for GrapplePlugin {
    fn default() -> Self { Self::new() }
}

impl QuartzPlugin for GrapplePlugin {
    fn name(&self) -> &str { "grapple" }

    /// Handle grapple commands dispatched via Action::PluginCall.
    fn on_call(&mut self, canvas: &mut Canvas, payload: &dyn std::any::Any) -> bool {
        if let Some(cmd) = payload.downcast_ref::<GrappleCommand>() {
            match cmd.clone() {
                GrappleCommand::Attach { target, grapple } => {
                    for name in canvas.store.get_names(&target) {
                        self.attach_grapple_impl(canvas, &name, grapple.clone());
                    }
                    true
                }
                GrappleCommand::Release { target } => {
                    for name in canvas.store.get_names(&target) {
                        self.grapple_constraints.remove(&name);
                    }
                    true
                }
                GrappleCommand::SetLength { target, value } => {
                    for name in canvas.store.get_names(&target) {
                        if let Some(g) = self.grapple_constraints.get_mut(&name) {
                            g.length = value.max(1.0);
                        }
                    }
                    true
                }
                GrappleCommand::SetStiffness { target, value } => {
                    for name in canvas.store.get_names(&target) {
                        if let Some(g) = self.grapple_constraints.get_mut(&name) {
                            g.stiffness = value.clamp(0.0, 1.0);
                        }
                    }
                    true
                }
                GrappleCommand::SetDamping { target, value } => {
                    for name in canvas.store.get_names(&target) {
                        if let Some(g) = self.grapple_constraints.get_mut(&name) {
                            g.damping = value.clamp(0.0, 1.0);
                        }
                    }
                    true
                }
                GrappleCommand::SetAnchor { target, x, y } => {
                    for name in canvas.store.get_names(&target) {
                        if let Some(g) = self.grapple_constraints.get_mut(&name) {
                            g.anchor = (x, y);
                            g.anchor_object = None;
                        }
                    }
                    true
                }
                GrappleCommand::SetAnchorObject { target, anchor_object } => {
                    for name in canvas.store.get_names(&target) {
                        if let Some(g) = self.grapple_constraints.get_mut(&name) {
                            g.anchor_object = Some(anchor_object.clone());
                        }
                    }
                    true
                }
                GrappleCommand::SetSwingBias { target, bias } => {
                    for name in canvas.store.get_names(&target) {
                        if let Some(g) = self.grapple_constraints.get_mut(&name) {
                            g.swing_bias = bias;
                        }
                    }
                    true
                }
            }
        } else {
            false
        }
    }

    /// Enforce all active grapple constraints after the physics step.
    fn on_post_solve(&mut self, canvas: &mut Canvas, _dt: f32) {
        self.enforce_grapple_constraints(canvas);
    }
}

impl GrapplePlugin {
    /// Attach a grapple constraint to a named game object (internal).
    fn attach_grapple_impl(&mut self, canvas: &mut Canvas, name: &str, mut grapple: GrappleConstraint) {
        // If anchor_object is set, resolve its current position as initial anchor
        if let Some(anchor_name) = &grapple.anchor_object {
            if let Some(anchor_obj) = canvas.get_game_object(anchor_name) {
                grapple.anchor = (
                    anchor_obj.position.0 + anchor_obj.size.0 * 0.5,
                    anchor_obj.position.1 + anchor_obj.size.1 * 0.5,
                );
            }
        }
        self.grapple_constraints.insert(name.to_string(), grapple);
        // Wake the body so the grapple takes effect immediately
        canvas.wake_body(name);
    }

    /// Enforce grapple constraints by applying position/velocity corrections
    /// directly to the store. Called AFTER the physics solver step so
    /// corrections override the solver's output (XPBD-style).
    fn enforce_grapple_constraints(&mut self, canvas: &mut Canvas) {
        if self.grapple_constraints.is_empty() {
            return;
        }

        // First, update anchors for grapples attached to objects
        let anchor_updates: Vec<(String, (f32, f32))> = self.grapple_constraints.iter()
            .filter_map(|(name, grapple)| {
                let anchor_name = grapple.anchor_object.as_ref()?;
                let anchor_obj = canvas.store.name_to_index.get(anchor_name.as_str())
                    .and_then(|&idx| canvas.store.objects.get(idx))?;
                Some((name.clone(), (
                    anchor_obj.position.0 + anchor_obj.size.0 * 0.5,
                    anchor_obj.position.1 + anchor_obj.size.1 * 0.5,
                )))
            })
            .collect();

        for (name, anchor_pos) in anchor_updates {
            if let Some(g) = self.grapple_constraints.get_mut(&name) {
                g.anchor = anchor_pos;
            }
        }

        // Solve each grapple and collect corrections
        struct GrappleCorr {
            idx: usize,
            position: Option<(f32, f32)>,
            velocity: Option<(f32, f32)>,
        }
        let mut corrections: Vec<GrappleCorr> = Vec::new();

        let names: Vec<String> = self.grapple_constraints.keys().cloned().collect();
        for name in &names {
            let idx = match canvas.store.name_to_index.get(name.as_str()) {
                Some(&i) => i,
                None => continue,
            };
            let obj = match canvas.store.objects.get(idx) {
                Some(o) => o,
                None => continue,
            };
            let obj_center = (
                obj.position.0 + obj.size.0 * 0.5,
                obj.position.1 + obj.size.1 * 0.5,
            );
            let obj_vel = obj.momentum;
            let half_w = obj.size.0 * 0.5;
            let half_h = obj.size.1 * 0.5;

            if let Some(grapple) = self.grapple_constraints.get_mut(name.as_str()) {
                let correction = grapple.solve(obj_center, obj_vel);
                if correction.applied() {
                    // Convert center position back to top-left corner
                    corrections.push(GrappleCorr {
                        idx,
                        position: correction.position.map(|(cx, cy)| (cx - half_w, cy - half_h)),
                        velocity: correction.velocity,
                    });
                }
            }
        }

        // Apply corrections to store
        for corr in corrections {
            if let Some(obj) = canvas.store.objects.get_mut(corr.idx) {
                if let Some(pos) = corr.position {
                    obj.position = pos;
                    canvas.layout.offsets[corr.idx] = pos;
                }
                if let Some(vel) = corr.velocity {
                    obj.momentum = vel;
                }
            }
        }
    }
}
