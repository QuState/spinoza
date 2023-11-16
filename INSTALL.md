# Install

## macOS

If you have `ld` or `linking with cc` errors when building on macOS,
the solution is to add the following sections to your global `cargo` 
config located in `~/.cargo`

If you don't have one yet:

```bash
touch ~/.cargo/config
```

In your editor of choice, add the following sections:
```toml
[target.x86_64-apple-darwin]
rustflags = [
  "-C", "link-arg=-undefined",
  "-C", "link-arg=dynamic_lookup",
]

[target.aarch64-apple-darwin]
rustflags = [
  "-C", "link-arg=-undefined",
  "-C", "link-arg=dynamic_lookup",
]
```