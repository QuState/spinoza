# CONTRIBUTING

Contributions are welcome, and they are greatly appreciated!

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs [here](https://github.com/QuState/spinoza/issues).

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged as `bug` and `help wanted` is up for grabs.

### Implement Features

Look through the GitHub issues for issues tagged as `enhancement`.

### Write Documentation

Spinoza could always use more documentation, whether as part of the official
docs, or even on personal blogs, articles, etc. Documentation is always greatly appreciated!

### Submit Feedback

The best way to send feedback is to raise an issue [here](https://github.com/QuState/spinoza/issues).

### Improve Performance

Spinoza strives to be a cutting-edge, high-performance simulator. We welcome and highly value 
any contributions that lead to measurable improvements in performance. Your efforts to enhance 
Spinoza's capabilities are greatly appreciated!

## Get Started!
Ready to contribute? Here's how to setup Spinoza for local development.

1. Fork the Spinoza repository
2. Clone your fork to your dev machine/environment:
```bash
git clone git@github.com:<username>/spinoza.git
```
3. [Install Rust](https://www.rust-lang.org/tools/install) and setup [nightly](https://rust-lang.github.io/rustup/concepts/channels.html) Rust

4. Setup the git hooks by going in your local Spinoza repo:
```bash
cd spinoza
git config core.hooksPath ./hooks 
```

5. When you're done with your changes, ensure the tests pass with:
```bash
cargo test
```

7. Commit your changes and push them to GitHub

8. Submit a pull request (PR) through the [GitHub website](https://github.com/QuState/spinoza/pulls).

## Pull Request Guidelines

Before you submit a pull request, please check the following:
- The pull request should include tests if it adds and/or changes functionalities.