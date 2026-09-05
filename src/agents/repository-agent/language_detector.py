import os


LANGUAGE_EXTENSIONS = {
    ".py": "python",

    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",

    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",

    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",

    ".cs": "c_sharp",

    ".go": "go",

    ".rs": "rust",

    ".rb": "ruby",

    ".php": "php",

    ".swift": "swift",

    ".dart": "dart",

    ".r": "r",

    ".sh": "bash",

    ".sql": "sql",
}


def detect_language(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    return LANGUAGE_EXTENSIONS.get(extension)