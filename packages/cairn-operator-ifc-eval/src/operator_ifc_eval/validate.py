"""IFC snippet validation (cairn-extensions#12): does a generated snippet parse, and does it hold
to the targeted IFC schema?

A snippet is what a model writes when asked for IFC: STEP entity instances (ISO 10303-21), such as
``#1=IFCPROJECT('3MDa0qu6X5eBFeTDmhEULg',$,'Sample Project',$,$,$,$,$,$);``, often inside a
fenced code block and surrounded by prose, sometimes as a whole file with its header. Each one is
checked in three steps, the same for every row and every model:

1. **Extract** (:func:`extract`). The fenced blocks when there are any; else the ``DATA;`` section
   of a whole file; else the text from the first ``#<n>=`` to the last ``;`` after it. Nothing
   found is a parse failure (``no IFC instance found``).
2. **Parse.** A strict reading of the instance statements (:func:`statements`): every statement is
   ``#<id>=<ENTITY>(<attributes>);`` with balanced parentheses and closed strings, and no id is
   used twice. Then IfcOpenShell reads the snippet, wrapped in a minimal header that names the
   schema, and every entity must be one the schema declares and every instance must come back.
   IfcOpenShell's own reader is lenient (it drops a statement it cannot read and keeps going), so
   the strict reading and the round trip are what make a snippet "parse"; IfcOpenShell is what
   knows the schema.
3. **Schema** (only for a snippet that parses): each instance has the number of attributes its
   entity declares; every ``#<id>`` it references is defined in the snippet; and
   ``ifcopenshell.validate`` finds no wrong attribute type, missing required attribute or bad
   aggregate. WHERE rules and global rules are not checked (``ifcopenshell.validate`` does not
   implement them).

The rates: ``parse_valid_rate`` = snippets that parse / snippets; ``schema_error_rate`` = parsed
snippets with at least one schema error / parsed snippets (``None`` when none parsed);
``valid_rate`` = snippets that parse with no schema error / snippets.

IfcOpenShell is the optional ``[ifc]`` extra, imported only here and only when a check runs:
without it :func:`available` is false and the callers report ``checked: false``, never a rate.
Nothing here gates: the rates are measurements for a person or a pipeline predicate to read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: The schema a check targets when none is named.
DEFAULT_SCHEMA = "IFC4"
#: Messages kept per row (a broken snippet can produce hundreds).
MAX_ERRORS_PER_ROW = 5

_HEADER = ("ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\nFILE_NAME('','',(''),(''),'','','');\n"
           "FILE_SCHEMA(('{schema}'));\nENDSEC;\nDATA;\n{data}\nENDSEC;\nEND-ISO-10303-21;\n")
_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.S)
_INSTANCE = re.compile(r"#\d+\s*=")
_DATA = re.compile(r"\bDATA\s*;(.*?)\bENDSEC\s*;", re.S)
_ID = re.compile(r"#(\d+)\s*=\s*")
_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


class SnippetSyntaxError(ValueError):
    """A statement that is not ``#<id>=<ENTITY>(<attributes>);``."""


@dataclass
class Statement:
    id: int
    entity: str
    arguments: int          # top-level attribute values
    references: list[int]   # every #<id> in the attribute values


@dataclass
class RowResult:
    parse_valid: bool
    schema_errors: list[str] = field(default_factory=list)
    error: str = ""          # why it did not parse
    instances: int = 0

    @property
    def valid(self) -> bool:
        return self.parse_valid and not self.schema_errors


def available() -> bool:
    """Is IfcOpenShell (the ``[ifc]`` extra) importable?"""
    try:
        import ifcopenshell  # noqa: F401, PLC0415
        import ifcopenshell.validate  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def version() -> str:
    import ifcopenshell  # noqa: PLC0415

    return str(getattr(ifcopenshell, "version", "?"))


def resolve_schema(name: str):
    """IfcOpenShell's schema for ``name`` (``IFC2X3``, ``IFC4``, ``IFC4X3_ADD2``; any case), or
    ``ValueError`` naming the ones it knows."""
    import ifcopenshell.ifcopenshell_wrapper as wrapper  # noqa: PLC0415

    try:
        return wrapper.schema_by_name(str(name or DEFAULT_SCHEMA).strip())
    except RuntimeError as e:
        known = [n for n in wrapper.schema_names() if n.upper().startswith("IFC")]
        raise ValueError(f"unknown IFC schema {name!r}; IfcOpenShell {version()} knows {known}") from e


def extract(text: str) -> str:
    """The IFC instances in a model's answer (see the module docstring); ``""`` when there are none."""
    text = str(text or "")
    blocks = _FENCE.findall(text)
    if blocks:
        text = "\n".join(blocks)
    data = _DATA.search(text)
    if data:
        return data.group(1).strip()
    first = _INSTANCE.search(text)
    if not first:
        return ""
    last = text.rfind(";")
    return text[first.start():last + 1].strip() if last > first.start() else text[first.start():].strip()


def _skip_comment(s: str, i: int) -> int:
    end = s.find("*/", i + 2)
    if end < 0:
        raise SnippetSyntaxError("unclosed comment")
    return end + 2


def statements(data: str) -> list[Statement]:
    """Read ``data`` strictly as instance statements, or raise :class:`SnippetSyntaxError`."""
    out: list[Statement] = []
    seen: set[int] = set()
    i, n = 0, len(data)
    while True:
        while i < n and (data[i].isspace() or data.startswith("/*", i)):
            i = _skip_comment(data, i) if data.startswith("/*", i) else i + 1
        if i >= n:
            break
        m = _ID.match(data, i)
        if not m:
            raise SnippetSyntaxError(f"expected '#<id>=' at {data[i:i + 30]!r}")
        ident = int(m.group(1))
        if ident in seen:
            raise SnippetSyntaxError(f"#{ident} is defined twice")
        seen.add(ident)
        i = m.end()
        m = _NAME.match(data, i)
        if not m:
            raise SnippetSyntaxError(f"#{ident}: expected an entity name and '(' at {data[i:i + 30]!r}")
        entity = m.group(1)
        i = m.end()
        depth, args, refs, empty = 1, 0, [], True
        while depth:
            if i >= n:
                raise SnippetSyntaxError(f"#{ident}={entity}: unbalanced parentheses")
            c = data[i]
            if c == "'":
                end = i + 1
                while True:
                    end = data.find("'", end)
                    if end < 0:
                        raise SnippetSyntaxError(f"#{ident}={entity}: unclosed string")
                    if data.startswith("''", end):
                        end += 2
                        continue
                    break
                i, empty = end + 1, False
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == "," and depth == 1:
                args += 1
            elif c == "#":
                ref = re.match(r"#(\d+)", data[i:])
                if ref:
                    refs.append(int(ref.group(1)))
            elif data.startswith("/*", i):
                i = _skip_comment(data, i)
                continue
            if depth and not c.isspace():
                empty = False
            i += 1
        while i < n and data[i].isspace():
            i += 1
        if i >= n or data[i] != ";":
            raise SnippetSyntaxError(f"#{ident}={entity}: missing ';' after the attributes")
        i += 1
        out.append(Statement(id=ident, entity=entity, arguments=0 if empty else args + 1, references=refs))
    if not out:
        raise SnippetSyntaxError("no IFC instance found")
    return out


def check(text: str, schema: str = DEFAULT_SCHEMA) -> RowResult:
    """One answer: extract, parse, and (when it parses) check it against ``schema``. Needs the
    ``[ifc]`` extra; :func:`available` says whether it is installed."""
    import ifcopenshell  # noqa: PLC0415
    import ifcopenshell.validate  # noqa: PLC0415

    target = resolve_schema(schema)
    data = extract(text)
    if not data:
        return RowResult(parse_valid=False, error="no IFC instance found")
    try:
        stmts = statements(data)
    except SnippetSyntaxError as e:
        return RowResult(parse_valid=False, error=str(e))
    declared: dict[int, int] = {}
    for s in stmts:
        try:
            decl = target.declaration_by_name(s.entity).as_entity()
        except RuntimeError:
            decl = None
        if decl is None:
            return RowResult(parse_valid=False, error=f"#{s.id}: {s.entity} is not an entity of {target.name()}")
        declared[s.id] = decl.attribute_count()
    try:
        model = ifcopenshell.file.from_string(_HEADER.format(schema=target.name(), data=data))
        read = {inst.id() for inst in model}
    except Exception as e:  # noqa: BLE001 — anything the reader refuses is a parse failure
        return RowResult(parse_valid=False, error=f"IfcOpenShell could not read it: {type(e).__name__}: {e}")
    lost = sorted(set(declared) - read)
    if lost:
        return RowResult(parse_valid=False, error=f"IfcOpenShell did not read #{lost[0]}"
                                                  + (f" and {len(lost) - 1} more" if len(lost) > 1 else ""))
    errors: list[str] = []
    for s in stmts:
        if s.arguments != declared[s.id]:
            errors.append(f"#{s.id}={s.entity}: {s.arguments} attributes, {target.name()} declares "
                          f"{declared[s.id]}")
        for ref in s.references:
            if ref not in declared:
                errors.append(f"#{s.id}={s.entity}: references #{ref}, which the snippet does not define")
    logger = ifcopenshell.validate.json_logger()
    try:
        ifcopenshell.validate.validate(model, logger)
    except Exception as e:  # noqa: BLE001 — a validator crash is reported on the row, never raised
        errors.append(f"validation failed: {type(e).__name__}: {e}")
    for st in logger.statements:
        message = " ".join(str(st.get("message") or "").split())
        where = [p for p in (_instance_label(st.get("instance")), st.get("attribute")) if p]
        errors.append(": ".join([*map(str, where), message]))
    return RowResult(parse_valid=True, schema_errors=errors[:MAX_ERRORS_PER_ROW]
                     + ([f"… and {len(errors) - MAX_ERRORS_PER_ROW} more"] if len(errors) > MAX_ERRORS_PER_ROW else []),
                     instances=len(stmts))


def _instance_label(inst) -> str:
    """``#1=IfcProject`` for an instance (its full text can run to kilobytes)."""
    if inst is None:
        return ""
    try:
        return f"#{inst.id()}={inst.is_a()}"
    except Exception:  # noqa: BLE001 — a label is best effort
        return str(inst)[:80]


@dataclass
class Summary:
    rows: int
    parse_valid: int
    schema_clean: int        # parsed with no schema error
    failures: list[dict]     # {row, stage: parse|schema, errors}

    @property
    def parse_valid_rate(self) -> float | None:
        return self.parse_valid / self.rows if self.rows else None

    @property
    def schema_error_rate(self) -> float | None:
        return (self.parse_valid - self.schema_clean) / self.parse_valid if self.parse_valid else None

    @property
    def valid_rate(self) -> float | None:
        return self.schema_clean / self.rows if self.rows else None


def summarize(texts: list[str], schema: str = DEFAULT_SCHEMA) -> Summary:
    """Every answer in ``texts`` checked, the rates and the per-row failures (rows numbered from 0)."""
    results = [check(t, schema) for t in texts]
    failures = []
    for i, r in enumerate(results):
        if not r.parse_valid:
            failures.append({"row": i, "stage": "parse", "errors": [r.error]})
        elif r.schema_errors:
            failures.append({"row": i, "stage": "schema", "errors": list(r.schema_errors)})
    return Summary(rows=len(results), parse_valid=sum(r.parse_valid for r in results),
                   schema_clean=sum(r.valid for r in results), failures=failures)


__all__ = ["DEFAULT_SCHEMA", "MAX_ERRORS_PER_ROW", "RowResult", "SnippetSyntaxError", "Statement", "Summary",
           "available", "check", "extract", "resolve_schema", "statements", "summarize", "version"]
