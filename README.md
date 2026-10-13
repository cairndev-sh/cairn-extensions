# cairn extensions

First-party public extensions for [cairn](https://github.com/cairndev-sh/cairn): operator
packs, the domain contracts they implement, and domain packs.

Each extension ships as its own wheel (`cairn-operator-*`, `cairn-contracts-*`,
`cairn-pack-*`) and declares the cairn engine versions it supports, so it installs and
releases independently of the engine. The engine discovers installed extensions through
entry points; nothing here is imported by the engine itself.

The engine, the builtin operator packs and the core contracts live in
[cairndev-sh/cairn](https://github.com/cairndev-sh/cairn).

## Status

Empty for now. Extensions land here as they are opened up.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
