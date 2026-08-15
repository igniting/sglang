# Appendix B — Environment Variables

Environment variables in SGLang are not read with `os.getenv` scattered through the
codebase. They are **declared**, in one file, as typed descriptors.

`python/sglang/srt/environ.py:37` `EnvField` is the base:

```python
class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name
```

`__set_name__` gives each descriptor the name of the attribute it was assigned to, so the
declaration `SGLANG_ENABLE_X = EnvBool(False)` sets the variable name from the attribute —
declaration and name cannot drift apart.

The assertion is the interesting part. `a = envs.A` would bind the *descriptor*, not its
value, and then `a` would silently never reflect the environment. The flag turns a subtle
bug into a loud one at import time.

`_resolve_default` supports callable defaults for platform-dependent values:

```python
    def _resolve_default(self) -> Any:
        # Support a callable default for lazily/platform-computed defaults
        # (e.g. EnvBool(_default_hip)); evaluated only when the env is unset.
        return self.default() if callable(self.default) else self.default
```

## The typed subclasses

| Class | Line | Parses |
| --- | --- | --- |
| `EnvBool` | `:134` | `1`/`true`/`yes` and friends |
| `EnvInt` | `:144` | integer |
| `EnvFloat` | `:187` | float |
| `EnvStr` | `:119` | string |
| `EnvTuple` | `:114` | comma-separated |
| `EnvJSON` | `:124` | JSON |

Typing matters more than it looks. `SGLANG_FOO=0` read with `os.getenv` is the truthy string
`"0"`; read through `EnvBool` it is `False`. That class of bug is why the indirection exists.

## Deprecation

`:152` `_DeprecatedEnvFallback` handles renames:

```python
class _DeprecatedEnvFallback:
    """Mixin for EnvField subclasses: if the canonical env var is not set,
    check *deprecated_name* and emit DeprecationWarning before reading it.

    Usage:
        SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(True, deprecated_name="SGLANG_NSA_FUSE_TOPK")
    """

    def get(self) -> Any:
        if os.getenv(self.name) is None:
            fallback = os.getenv(self.deprecated_name)
            if fallback is not None:
                warnings.warn(
                    f"Environment variable '{self.deprecated_name}' is deprecated; "
                    f"use '{self.name}' instead. ...",
                    DeprecationWarning,
                    stacklevel=2,
                )
                os.environ[self.name] = fallback
        return super().get()
```

The old name still works, with a warning, and the new name takes precedence. `:179`
`EnvBoolWithAlias` and `:183` `EnvIntWithAlias` are the concrete forms. This is how the
legacy `SGL_*` prefix was migrated to `SGLANG_*` without breaking deployments.

## Conventions

`.claude/skills/env-var-conventions/SKILL.md` is the authority. In brief:

- **Declare in `python/sglang/srt/environ.py`.** Never `os.getenv` at a use site.
- **Prefix `SGLANG_`.**
- **Access through `envs.NAME.get()`**, not by string.
- **Deprecate with an alias class**, never by deleting.

Chapter 12 showed a use in context:

```python
        if envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.get():
            return self._load_weights_v2(weights)
        return self._legacy_load_weights(weights)
```

That is the pattern throughout: an environment variable gating an alternative
implementation during a migration.

## Where they appear in this book

- `SGLANG_ENABLE_WEIGHT_LOADER_V2` — Chapter 12, the loader migration.
- `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY` — Chapter 5, per-iteration invariant checks.
- `SGLANG_DEBUG_MEMORY_POOL` — Chapter 9, allocator assertions.

The general shape: environment variables are for things an *operator* should not normally
touch — debug modes, migration gates, platform workarounds. Anything a user should tune is a
server argument (Appendix A) instead.

`docs/docs/references/environment_variables.mdx` is the generated reference and stays
closer to the source than this appendix.
