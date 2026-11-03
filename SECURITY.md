# Security policy

wvc reads checkpoints and stores that may come from people you do not trust: a Hub download, an unpacked archive, a cloned or pulled `.wvc` directory. The design goal is that such input can make wvc refuse, never make it write outside the directory you named, run code, or report bytes as verified that are not. If you find a way around that, please report it privately first.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on this repository (the "Security" tab, then "Report a vulnerability"). That opens a private advisory that only the maintainer sees. If you cannot use it, open an issue that says only that you have a security report, and the maintainer will arrange a private channel.

Please include the wvc version (`wvc --version`), the command, and a way to reproduce: a small store or checkpoint, or the script that builds one. Do not attach real model weights or anything you are not allowed to share.

This is a one-person project. You will get an acknowledgement within seven days and a fix or a reasoned answer within thirty. Credit goes in the changelog unless you ask otherwise.

## What counts

In scope, for any input wvc accepts:

- A write outside the checkout directory, or inside `.wvc` where a record does not point.
- Code execution, including through a file name, a record, or a Hub listing.
- A checkout, `verify`, or `fsck` that reports success for bytes that do not match the recorded hashes.
- Unbounded memory, disk, or time from a small input: the limits in [docs/spec-v0.md](docs/spec-v0.md) section 3.6 are the contract, and a way past them is a bug.
- Error output that reproduces file contents or terminal control sequences.

Known limits that are documented rather than defended, and are not reports:

- A tensor under the configured `max_tensor_bytes` ceiling (16 GiB by default) is decoded in memory; a checkout is bounded by free disk space, not by a smaller cap.
- Verification markers under `.wvc/verified/` are trusted in a store copied as a directory; a pulled store never receives them.
- `wvc bisect --test` runs the command you give it through your shell.
- Nothing in wvc inspects what a model does; it records and verifies bytes.

## Supported versions

Only the latest release on the `main` branch receives fixes. Pre-release versions (`0.x`) may change behaviour between releases; the changelog says when a fix changes what a command accepts.
