# Starframe

## What

In disc golf, a starframe occurs when every player in a group scores a birdie
on the same hole.

This starframe, however, is my personal game engine written in Rust.
Its main feature is the physics engine,
with design driven by sidescrolling action games.

Stuff made with it:
- [animated artworks][art]
- [Velgi] — Fangame of Velgress from UFO 50
- [Flamegrower] — Puzzle platformer about vines and fire (WIP, progressing very slowly)

## Screenshots

![A stack of boxes and some colored balls, lit by a setting sun from the side.
The stack of boxes blocks the sunlight, leaving a blue shadow.](screenshots/sunset.jpg)

![Various shapes scattered on the ground, some of them emitting light
and others casting colored shadows.
There are a few trees in the background.](screenshots/night.jpg)

![Various shapes suspended on top of a rope,
casting colored shadows underneath them.](screenshots/day.jpg)

## Features

Most links in the following are to blog posts discussing the feature.

- 2D rigid body and particle physics
  - [solver][blog-constraints] based on [Extended Position-Based Dynamics][xpbd]
  - [collider shapes][blog-colliders]: circles, convex polygons, and rounded convex polygons
    - compound shapes of many of these also supported
  - [raycasts and spherecasts][blog-colliders]
  - [particle-based ropes][blog-ropes] with full coupling with rigid bodies
- [graphics][blog-graphics] using [wgpu]
  - global illumination with Radiance Cascades
  - textured and animated triangle meshes, optionally with normal maps

## Who this is for

The more I work on graphics, the more it becomes clear
that this is an _extremely_ opinionated engine
for my personal style of making games and graphics.
I don't have the time or interest to support use cases other than mine.
So this is a thing for personal use first and foremost.
If you want to try making something with my very specific graphics workflow,
or just use the physics part and ignore the rest, be my guest.
I can't promise it will work out well.

## Blog

I write about this project once in a blue moon on [my website](https://molentum.me/blog/).

## Sandbox example

I have a little sandbox I use for testing new features where you can throw
blocks around with the mouse and move a rudimentary platformer character that
shoots some rather heavy bullets. Here's how you can check it out:

### The manual way

1. Install [Rust](https://www.rust-lang.org/learn/get-started)
2. You may need to install `pkgconfig` and drivers for Vulkan, DX12, or Metal
   depending on your platform
3. Clone and navigate to this repository
4. `cargo run --example sandbox`

### The easy way, using [Nix](https://nixos.org/nix/) (on Linux)

1. Clone and navigate to this repository
2. `nix-shell`
3. `cargo run --example sandbox`

### Keybindings

Disclaimer: these might be out of date - the sandbox changes in quick and
dirty ways

```text
Space   - step one frame while paused

Left mouse   - grab objects
Middle mouse - move the camera
Mouse wheel  - zoom the camera

Arrows  - move the player
LShift  - jump
Z       - shoot
```

[xpbd]: https://matthias-research.github.io/pages/publications/PBDBodies.pdf
[wgpu]: https://github.com/gfx-rs/wgpu-rs
[blog-constraints]: https://molentum.me/blog/2021/starframe-constraints/
[blog-ropes]: https://molentum.me/blog/2021/starframe-ropes/
[blog-colliders]: https://molentum.me/blog/2025/rounding-collider-corners/
[blog-graphics]: https://molentum.me/blog/2025/game-graphics/
[flamegrower]: https://github.com/m0lentum/flamegrower
[art]: https://github.com/m0lentum/art
[flamegrower]: https://github.com/m0lentum/flamegrower
[velgi]: https://molentum.itch.io/velgi
