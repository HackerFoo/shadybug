[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/HackerFoo/shadybug#license)
[![Crates.io](https://img.shields.io/crates/v/shadybug.svg)](https://crates.io/crates/shadybug)
[![Downloads](https://img.shields.io/crates/d/shadybug.svg)](https://crates.io/crates/shadybug)
[![Docs](https://docs.rs/shadybug/badge.svg)](https://docs.rs/bevy/latest/shadybug/)

Shadybug
========

Shadybug is a simple reference software renderer to be used for debugging shaders.

It's designed to make it easy to port shaders to Rust, and run them for one triangle or four pixels.

Examples
--------

Try this:

    cargo run --example cube  --features image

which should create an image named `cube.png`:

![a red cube](images/cube.png)
