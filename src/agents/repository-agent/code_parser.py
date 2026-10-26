from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tree_sitter import Parser
from tree_sitter_languages import get_language

from language_detector import detect_language


SUPPORTED_EXTENSIONS = {
    ".py",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".kt",
    ".kts",
    ".swift",
    ".scala",
    ".dart",
    ".r",
    ".sh",
    ".bash",
    ".sql",
}


IGNORED_DIRECTORIES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "node_modules",
    "target",
    "build",
    "dist",
    "venv",
    ".venv",
    "env",
    ".env",
}


SUPPORTED_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "tsx",
    "java",
    "kotlin",
    "scala",
    "c",
    "cpp",
    "c_sharp",
    "csharp",
    "go",
    "rust",
    "ruby",
    "php",
    "swift",
    "dart",
    "r",
    "bash",
    "shell",
    "sql",
}


def _normalize_language(language: str) -> str:
    """Normalize language detector output to tree-sitter names."""

    normalized = (language or "").strip().lower()

    aliases = {
        "js": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "c++": "cpp",
        "cxx": "cpp",
        "cc": "cpp",
        "c#": "c_sharp",
        "csharp": "c_sharp",
        "cs": "c_sharp",
        "sh": "bash",
        "shellscript": "bash",
        "golang": "go",
        "py": "python",
        "rb": "ruby",
    }

    return aliases.get(normalized, normalized)

def create_parser(language_name: str) -> Parser:
    """Create a parser using tree-sitter 0.20.x."""

    try:
        language = get_language(language_name)
    except Exception as error:
        raise RuntimeError(
            f"Could not load tree-sitter language "
            f"{language_name!r}: {error}"
        ) from error

    parser = Parser()
    parser.set_language(language)
    return parser



def get_source_files(repo_path: str) -> list[str]:
    """Return supported source files from a repository."""

    root = Path(repo_path).resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Repository path does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Repository path is not a directory: {root}"
        )

    files: list[str] = []

    print(
        f"[repository_agent] scanning source files under {root}",
        flush=True,
    )

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative_parts = path.relative_to(root).parts

        # Skip .git, build folders, virtual environments, etc.
        if any(
            directory in IGNORED_DIRECTORIES
            for directory in relative_parts
        ):
            continue

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        files.append(str(path))

    files.sort()

    print(
        f"[repository_agent] get_source_files returning "
        f"{len(files)} file(s)",
        flush=True,
    )

    for file_path in files[:50]:
        print(
            f"[repository_agent] source file: {file_path}",
            flush=True,
        )

    return files


def find_identifier(node: Any) -> Any | None:
    """Find an identifier inside a tree-sitter node."""

    if node.type in {
        "identifier",
        "field_identifier",
        "property_identifier",
        "type_identifier",
        "namespace_identifier",
    }:
        return node

    for child in node.children:
        result = find_identifier(child)

        if result is not None:
            return result

    return None


def _decode_node_text(
    node: Any,
    source_code: bytes,
) -> str:
    return source_code[
        node.start_byte:node.end_byte
    ].decode(
        "utf-8",
        errors="replace",
    )


def get_node_name(
    node: Any,
    source_code: bytes,
) -> str:
    """Extract a function or method name from a tree-sitter node."""

    # Most grammars expose a `name` field.
    name_node = node.child_by_field_name("name")

    if name_node is not None:
        return _decode_node_text(
            name_node,
            source_code,
        )

    # C/C++ function definitions usually expose a declarator.
    declarator = node.child_by_field_name("declarator")

    if declarator is not None:
        name_node = find_identifier(declarator)

        if name_node is not None:
            return _decode_node_text(
                name_node,
                source_code,
            )

    # Some grammars use a direct child identifier.
    name_node = find_identifier(node)

    if name_node is not None:
        return _decode_node_text(
            name_node,
            source_code,
        )

    return "unknown"


def _classify_node(
    node_type: str,
    language_name: str,
    parent_name: str | None,
) -> str | None:
    """Map tree-sitter node types to repository-agent chunk types."""

    # Classes, interfaces, structs, and enums.
    if node_type in {
        "class_definition",
        "class_declaration",
        "interface_declaration",
        "struct_declaration",
        "enum_declaration",
        "enum_definition",
        "record_declaration",
    }:
        return "class"

    # Functions and methods across supported grammars.
    if node_type in {
        "function_definition",
        "function_declaration",
        "function_item",
        "function",
        "method_definition",
        "method_declaration",
        "method",
        "singleton_method",
        "constructor_declaration",
        "constructor_definition",
    }:
        method_languages = {
            "java",
            "kotlin",
            "scala",
            "c_sharp",
            "ruby",
            "php",
            "swift",
        }

        if node_type in {
            "method_definition",
            "method_declaration",
            "method",
            "singleton_method",
            "constructor_declaration",
            "constructor_definition",
        }:
            return "method"

        if language_name in method_languages and parent_name:
            return "method"

        return "function"

    return None


def extract_code_nodes(
    file_path: str,
    repository_path: str,
) -> list[dict[str, Any]]:
    """Extract functions, methods, and classes from one source file."""

    file_path = str(Path(file_path).resolve())
    repository_path = str(Path(repository_path).resolve())

    detected_language = detect_language(file_path)
    language_name = _normalize_language(detected_language)

    print(
        f"[repository_agent] parsing {file_path} "
        f"as language={language_name!r}",
        flush=True,
    )

    if language_name not in SUPPORTED_LANGUAGES:
        print(
            f"[repository_agent] unsupported detected language "
            f"{detected_language!r} for {file_path}",
            flush=True,
        )
        return []

    parser = create_parser(language_name)

    with open(file_path, "rb") as source_file:
        source_code = source_file.read()

    tree = parser.parse(source_code)

    nodes: list[dict[str, Any]] = []

    def traverse(
        node: Any,
        parent_name: str | None = None,
    ) -> None:
        node_type = node.type

        code_type = _classify_node(
            node_type=node_type,
            language_name=language_name,
            parent_name=parent_name,
        )

        current_name = parent_name

        if code_type is not None:
            current_name = get_node_name(
                node,
                source_code,
            )

            code = _decode_node_text(
                node,
                source_code,
            )

            try:
                relative_file = os.path.relpath(
                    file_path,
                    repository_path,
                )
            except ValueError:
                # Windows drives can produce a ValueError if paths differ.
                relative_file = file_path

            nodes.append(
                {
                    "file": relative_file.replace("\\", "/"),
                    "language": language_name,
                    "type": code_type,
                    "name": current_name,
                    "parent": parent_name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "code": code,
                }
            )

        for child in node.children:
            traverse(
                child,
                current_name,
            )

    traverse(tree.root_node)

    print(
        f"[repository_agent] parsed {file_path}: "
        f"{len(nodes)} node(s)",
        flush=True,
    )

    return nodes