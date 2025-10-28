"""Utility to assist migrating JSP views to Angular components.

This module implements a light-weight parser that identifies common JSP
constructs—imports, includes, inline styles, forms, and buttons—and converts
those elements into Angular-friendly equivalents.  The resulting migration
artifacts (Angular template HTML, component stylesheet, and TypeScript
component stub) can then be written to disk.

The tool is intentionally conservative: it focuses on recognizing well-known
JSP tags and scriptlets rather than attempting to fully interpret JSP logic.
When the parser encounters JSP elements it does not understand, it keeps the
original markup in the generated Angular template so that developers can review
and adapt it manually.
"""
from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Sequence, Tuple
from xml.etree import ElementTree as ET

JSP_DIRECTIVE_RE = re.compile(r"<%@\s*(.*?)%>", re.DOTALL)
JSP_INLINE_EXPRESSION_RE = re.compile(r"<%=\s*(.*?)\s*%>", re.DOTALL)
JSP_SCRIPTLET_RE = re.compile(r"<%(?![@=])(.*?)%>", re.DOTALL)


def parse_imports(content: str) -> Tuple[str, List[str]]:
    """Extract JSP directives (e.g. ``<%@ include file="…" %>``).

    Returns the content with directives removed, alongside a list of the raw
    directive strings.  The directives are preserved verbatim so that they can
    be surfaced to developers or translated into Angular-specific constructs at
    a later time.
    """

    directives: List[str] = []

    def _strip_directive(match: re.Match[str]) -> str:
        directive_body = match.group(1).strip()
        directives.append(directive_body)
        return ""

    without_directives = JSP_DIRECTIVE_RE.sub(_strip_directive, content)
    return without_directives, directives


@dataclass
class FormInfo:
    """Metadata captured for each form encountered in the JSP template."""

    original_name: str
    template_reference: str
    submit_handler: str


def _normalize_identifier(value: str, fallback: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")
    if not sanitized:
        sanitized = fallback
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _to_camel_case(value: str) -> str:
    parts = re.split(r"[^0-9a-zA-Z]+", value)
    return "".join(part.capitalize() for part in parts if part)


class JspToAngularParser(HTMLParser):
    """HTML parser that translates JSP tags into Angular-friendly markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.angular_template: List[str] = []
        self.styles: List[str] = []
        self.directives: List[str] = []
        self.includes: List[str] = []
        self.form_stack: List[FormInfo] = []
        self.forms: List[FormInfo] = []
        self.button_stack: List[str] = []
        self._collecting_style = False
        self._current_style_chunks: List[str] = []

    # ------------------------------------------------------------------
    # HTMLParser overrides
    # ------------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        if tag.lower() == "jsp:include":
            self._handle_include(attrs)
            return
        if tag.lower() == "style":
            self._collecting_style = True
            self._current_style_chunks = []
            return
        if tag.lower() == "form":
            self._handle_form_start(attrs)
            return
        if tag.lower() == "button":
            self._handle_button_start(attrs)
            return
        self.angular_template.append(self._reconstruct_tag(tag, attrs))

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        if tag.lower() == "jsp:include":
            self._handle_include(attrs)
            return
        if tag.lower() == "style":
            # Uncommon, but treat as inline styles.
            self._collecting_style = False
            style_content = self._render_style(attrs)
            if style_content:
                self.styles.append(style_content)
            return
        self.angular_template.append(self._reconstruct_tag(tag, attrs, is_self_closing=True))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "style":
            self._collecting_style = False
            style_content = "".join(self._current_style_chunks).strip()
            if style_content:
                self.styles.append(style_content)
            self._current_style_chunks = []
            return
        if tag.lower() == "form":
            self.angular_template.append("</form>")
            if self.form_stack:
                self.form_stack.pop()
            return
        if tag.lower() == "button":
            replacement = self.button_stack.pop() if self.button_stack else "button"
            if replacement == "form_button":
                self.angular_template.append("</button>")
            else:
                self.angular_template.append("</a>")
            return
        if tag.lower() == "jsp:include":
            return
        self.angular_template.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._collecting_style:
            self._current_style_chunks.append(data)
        else:
            converted = self._convert_jsp_expressions(data)
            self.angular_template.append(converted)

    def handle_comment(self, data: str) -> None:
        self.angular_template.append(f"<!--{data}-->")

    # ------------------------------------------------------------------
    # Handlers for specific tags
    # ------------------------------------------------------------------
    def _handle_include(self, attrs: Sequence[Tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        include_target = attr_dict.get("page") or attr_dict.get("file") or ""
        component_name = Path(include_target).stem if include_target else "include"
        normalized = component_name.replace("_", "-")
        angular_tag = f"app-{normalized}" if normalized else "app-include"
        self.includes.append(include_target)
        self.angular_template.append(f"<{angular_tag}></{angular_tag}>")

    def _handle_form_start(self, attrs: Sequence[Tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        original_name = attr_dict.get("name") or attr_dict.get("id") or f"form_{len(self.form_stack)+1}"
        template_reference = _normalize_identifier(f"{original_name}_form", "form")
        submit_handler = f"on{_to_camel_case(original_name) or 'Form'}Submit"
        form_info = FormInfo(
            original_name=original_name,
            template_reference=template_reference,
            submit_handler=submit_handler,
        )
        self.form_stack.append(form_info)
        self.forms.append(form_info)

        other_attrs = [
            (name, value)
            for name, value in attrs
            if name not in {"name", "id"}
        ]
        rendered_attrs = self._render_attributes(other_attrs)
        self.angular_template.append(
            f"<form #{template_reference}=\"ngForm\" "
            f"(ngSubmit)=\"{submit_handler}({template_reference})\"{rendered_attrs}>"
        )

    def _handle_button_start(self, attrs: Sequence[Tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        if self.form_stack:
            form_info = self.form_stack[-1]
            cleaned_attrs = [
                (name, value)
                for name, value in attrs
                if name not in {"type", "onclick"}
            ]
            rendered_attrs = self._render_attributes(cleaned_attrs)
            self.angular_template.append(
                f"<button type=\"submit\" (click)=\"{form_info.submit_handler}({form_info.template_reference})\"{rendered_attrs}>"
            )
            self.button_stack.append("form_button")
        else:
            target = attr_dict.get("data-link") or attr_dict.get("href") or "/"
            cleaned_attrs = [
                (name, value)
                for name, value in attrs
                if name not in {"type", "onclick", "href"}
            ]
            rendered_attrs = self._render_attributes(cleaned_attrs)
            self.angular_template.append(
                f"<a class=\"btn\" routerLink=\"{html.escape(target, quote=True)}\"{rendered_attrs}>"
            )
            self.button_stack.append("link")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _reconstruct_tag(
        self,
        tag: str,
        attrs: Sequence[Tuple[str, str | None]],
        *,
        is_self_closing: bool = False,
    ) -> str:
        rendered_attrs = self._render_attributes(attrs)
        trailing = " /" if is_self_closing else ""
        return f"<{tag}{rendered_attrs}{trailing}>"

    def _render_attributes(self, attrs: Sequence[Tuple[str, str | None]]) -> str:
        rendered = []
        for name, value in attrs:
            if value is None:
                rendered.append(f" {name}")
            else:
                escaped = html.escape(value, quote=True)
                rendered.append(f" {name}=\"{escaped}\"")
        return "".join(rendered)

    def _convert_jsp_expressions(self, text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            expression = match.group(1).strip()
            if not expression:
                return ""
            return f"{{{{ {expression} }}}}"

        converted = JSP_INLINE_EXPRESSION_RE.sub(_replace, text)

        def _wrap_scriptlet(match: re.Match[str]) -> str:
            code = match.group(1).strip()
            if not code:
                return ""
            return f"<!-- JSP scriptlet: {html.escape(code)} -->"

        converted = JSP_SCRIPTLET_RE.sub(_wrap_scriptlet, converted)
        return converted

    def _render_style(self, attrs: Sequence[Tuple[str, str | None]]) -> str:
        attr_dict = dict(attrs)
        inline_css = attr_dict.get("type", "")
        return inline_css.strip()


@dataclass
class MigrationArtifacts:
    """Container for the outputs generated during migration."""

    template_html: str
    stylesheet: str
    component_ts: str
    metadata_xml: str


def convert_content(content: str, component_selector: str) -> MigrationArtifacts:
    processed_content, directives = parse_imports(content)

    parser = JspToAngularParser()
    parser.directives.extend(directives)
    parser.feed(processed_content)
    parser.close()

    template = "".join(parser.angular_template)
    stylesheet = "\n\n".join(parser.styles)

    component_class_base = _to_camel_case(component_selector) or "Migrated"
    component_class_name = f"{component_class_base}Component"

    forms: List[dict[str, str]] = [
        {
            "original": info.original_name,
            "templateRef": info.template_reference,
            "handler": info.submit_handler,
        }
        for info in parser.forms
    ]

    component_ts = _render_component_ts(
        selector=component_selector,
        class_name=component_class_name,
        directives=parser.directives,
        includes=parser.includes,
        forms=forms,
    )

    metadata_xml = _build_metadata_xml(
        directives=parser.directives,
        includes=parser.includes,
        forms=forms,
    )
    return MigrationArtifacts(
        template_html=template,
        stylesheet=stylesheet,
        component_ts=component_ts,
        metadata_xml=metadata_xml,
    )


def _render_component_ts(
    *,
    selector: str,
    class_name: str,
    directives: Sequence[str],
    includes: Sequence[str],
    forms: Sequence[dict[str, str]],
) -> str:
    directive_summary = ", ".join(directives) if directives else "none"
    include_summary = ", ".join(includes) if includes else "none"
    metadata_lines = [
        "  // Migrated from JSP",
        f"  // Original directives: {directive_summary}",
        f"  // Includes: {include_summary}",
    ]

    form_methods = [
        _render_form_method(form_info)
        for form_info in forms
    ]
    if not form_methods:
        form_methods.append("  // TODO: add component logic here\n")

    metadata_block = "\n".join(metadata_lines)
    form_block = "".join(form_methods).rstrip()

    component_lines = [
        "import { Component } from '@angular/core';",
        "import { NgForm } from '@angular/forms';",
        "",
        "@Component({",
        f"  selector: 'app-{selector}',",
        f"  templateUrl: './{selector}.component.html',",
        f"  styleUrls: ['./{selector}.component.scss']",
        "})",
        f"export class {class_name} {{",
        metadata_block,
    ]
    if form_block:
        component_lines.append(form_block)
    component_lines.append("}")
    return "\n".join(line for line in component_lines if line is not None).strip("\n")


def _build_metadata_xml(
    *,
    directives: Sequence[str],
    includes: Sequence[str],
    forms: Sequence[dict[str, str]],
) -> str:
    root = ET.Element("migration")

    directives_el = ET.SubElement(root, "directives")
    for directive in directives:
        directive_el = ET.SubElement(directives_el, "directive")
        directive_el.text = directive

    includes_el = ET.SubElement(root, "includes")
    for include in includes:
        include_el = ET.SubElement(includes_el, "include")
        include_el.text = include

    forms_el = ET.SubElement(root, "forms")
    for form in forms:
        form_el = ET.SubElement(forms_el, "form")
        form_el.set("original", form["original"])
        form_el.set("templateRef", form["templateRef"])
        form_el.set("handler", form["handler"])

    _indent_xml(root)
    return ET.tostring(root, encoding="unicode")


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "  "
    children = list(element)
    if not children:
        return
    element.text = "\n" + indent * (level + 1)
    for i, child in enumerate(children):
        _indent_xml(child, level + 1)
        if i == len(children) - 1:
            child.tail = "\n" + indent * level
        else:
            child.tail = "\n" + indent * (level + 1)


def _render_form_method(form_info: dict[str, str]) -> str:
    handler = form_info["handler"]
    template_ref = form_info["templateRef"]
    original = form_info["original"]
    return (
        "  "
        + f"{handler}(form: NgForm): void {{\n"
        + f"    // Migrated from JSP form '{original}'\n"
        + "    if (form.valid) {\n"
        + "      // TODO: implement submit logic\n"
        + "    }\n"
        + "  }\n\n"
    )


def migrate_file(source: Path, destination_root: Path) -> MigrationArtifacts:
    if not source.exists():
        raise FileNotFoundError(f"JSP file '{source}' does not exist")

    content = source.read_text(encoding="utf-8")
    selector = source.stem.replace("_", "-")
    artifacts = convert_content(content, selector)

    component_dir = destination_root / selector
    component_dir.mkdir(parents=True, exist_ok=True)

    (component_dir / f"{selector}.component.html").write_text(
        artifacts.template_html,
        encoding="utf-8",
    )
    (component_dir / f"{selector}.component.scss").write_text(
        artifacts.stylesheet,
        encoding="utf-8",
    )
    (component_dir / f"{selector}.component.ts").write_text(
        artifacts.component_ts,
        encoding="utf-8",
    )
    (component_dir / f"{selector}.metadata.xml").write_text(
        artifacts.metadata_xml,
        encoding="utf-8",
    )
    return artifacts


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate JSP views to Angular component skeletons",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the JSP file to migrate",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Directory where Angular component files should be created",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print generated artifacts instead of writing files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.print:
        artifacts = convert_content(
            args.source.read_text(encoding="utf-8"),
            component_selector=args.source.stem.replace("_", "-"),
        )
        print("--- template (HTML) ---")
        print(artifacts.template_html)
        print("\n--- stylesheet (SCSS) ---")
        print(artifacts.stylesheet)
        print("\n--- component (TypeScript) ---")
        print(artifacts.component_ts)
        print("\n--- metadata (XML) ---")
        print(artifacts.metadata_xml)
        return 0

    args.destination.mkdir(parents=True, exist_ok=True)
    migrate_file(args.source, args.destination)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
