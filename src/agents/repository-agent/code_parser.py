import os

from tree_sitter import Parser
from tree_sitter_languages import get_language

from language_detector import detect_language


IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    ".idea",
    ".vscode",
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
    "go",
    "rust",
    "ruby",
    "php",
    "swift",
    "dart",
    "r",
    "bash",
    "sql"
}

def create_parser(language_name):

    language = get_language(language_name)

    parser = Parser()
    parser.set_language(language)

    return parser

def get_source_files(repository_path):

    source_files = []

    for root, dirs, files in os.walk(repository_path):

        dirs[:] = [
            directory
            for directory in dirs
            if directory not in IGNORED_DIRECTORIES
        ]

        for file in files:

            file_path = os.path.join(root, file)

            language = detect_language(file_path)

            if language in SUPPORTED_LANGUAGES:

                source_files.append(file_path)

    return source_files

def find_identifier(node):

    if node.type in {
        "identifier",
        "field_identifier",
    }:
        return node

    for child in node.children:

        result = find_identifier(child)

        if result is not None:
            return result

    return None

def get_node_name(node, source_code):

    # Normal case
    name_node = node.child_by_field_name("name")

    if name_node is not None:

        return source_code[
            name_node.start_byte:name_node.end_byte
        ].decode("utf-8", errors="ignore")

    # C / C++ function definition
    declarator = node.child_by_field_name("declarator")

    if declarator is not None:

        name_node = find_identifier(declarator)

        if name_node is not None:

            return source_code[
                name_node.start_byte:name_node.end_byte
            ].decode("utf-8", errors="ignore")

    return "unknown"


def extract_code_nodes(file_path, repository_path):

    language_name = detect_language(file_path)

    if language_name not in SUPPORTED_LANGUAGES:
        return []

    parser = create_parser(language_name)

    with open(file_path, "rb") as f:
        source_code = f.read()

    tree = parser.parse(source_code)

    nodes = []

    def traverse(node, parent_name=None):

        node_type = node.type
        code_type = None

        # Python
        if node_type in {
            "function_definition",
        }:
            code_type = "function"

        elif node_type == "class_definition":
            code_type = "class"

        # JavaScript / TypeScript
        elif node_type in {
            "function_declaration",
            "method_definition",
        }:
            code_type = "method" if parent_name else "function"

        elif node_type == "class_declaration":
            code_type = "class"

        # Java
        elif node_type == "class_declaration":
            code_type = "class"

        elif node_type in {
            "method_declaration",
        }:
            code_type = "method"

        # C / C++
        elif node_type == "function_definition":
            code_type = "function"

        # Go
        elif node_type in {
            "function_declaration",
            "method_declaration",
        }:
            code_type = "method" if parent_name else "function"

        # Rust
        elif node_type == "function_item":
            code_type = "function"

        # Ruby
        elif node_type in {
            "method",
            "singleton_method",
        }:
            code_type = "method"

        # PHP
        elif node_type in {
            "function_definition",
            "method_declaration",
        }:
            code_type = "method" if parent_name else "function"

        # C#
        elif node_type in {
            "method_declaration",
        }:
            code_type = "method"

        current_name = parent_name

        if code_type is not None:

            current_name = get_node_name(
                node,
                source_code
            )

            code = source_code[
                node.start_byte:node.end_byte
            ].decode(
                "utf-8",
                errors="ignore"
            )

            nodes.append({
                "file": os.path.relpath(
                    file_path,
                    repository_path
                ),
                "language": language_name,
                "type": code_type,
                "name": current_name,
                "parent": parent_name,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "code": code,
            })


        for child in node.children:

            traverse(
                child,
                current_name
            )

    traverse(tree.root_node)

    return nodes