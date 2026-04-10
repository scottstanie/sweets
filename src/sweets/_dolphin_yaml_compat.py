"""Side-effect import: teach dolphin's YamlModel walker about anyOf-of-refs.

dolphin's `_add_comments` JSON-schema walker assumes every entry inside an
``anyOf`` carries a primitive ``type`` key, but a Pydantic union of two
sub-models emits ``$ref``-only entries. We pre-walk the schema and inject
a synthetic ``type`` for those entries so the original walker keeps working.

Real fix is a 1-liner upstream in dolphin (see REVIVAL.md). This module
exists so the patch lives in one place and ``sweets.core`` can apply it
via a normal top-of-file import.
"""

from __future__ import annotations

import dolphin.workflows.config._yaml_model as _dyaml

_orig_add_comments = _dyaml._add_comments


def _patched_add_comments(
    loaded_yaml,  # noqa: ANN001
    schema: dict,
    indent: int = 0,
    definitions=None,  # noqa: ANN001
    indent_per_level: int = 2,
):
    for val in schema.get("properties", {}).values():
        if "anyOf" in val:
            for entry in val["anyOf"]:
                if "$ref" in entry and "type" not in entry:
                    entry["type"] = entry["$ref"].rsplit("/", 1)[-1]
    return _orig_add_comments(
        loaded_yaml,
        schema,
        indent=indent,
        definitions=definitions,
        indent_per_level=indent_per_level,
    )


_dyaml._add_comments = _patched_add_comments
