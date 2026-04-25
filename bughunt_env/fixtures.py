from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BugCase:
    name: str
    files: Dict[str, str]
    file_tree: List[str]
    failing_test: str
    stderr: str
    bug_file: str
    bug_line: int
    bug_summary: str
    symbols: Dict[str, str]
    callers: Dict[str, str]


def get_cases() -> List[BugCase]:
    return [
        BugCase(
            name="off_by_one_cart_total",
            files={
                "README.md": "# Cart Utils\\nSmall checkout helper library.\\n",
                "cart.py": (
                    "def calculate_total(items):\\n"
                    "    total = 0\\n"
                    "    for i in range(len(items) - 1):\\n"
                    "        total += items[i][\"price\"]\\n"
                    "    return total\\n"
                ),
                "tests/test_cart.py": (
                    "from cart import calculate_total\\n\\n"
                    "def test_calculate_total_counts_all_items():\\n"
                    "    items = [{\"price\": 10}, {\"price\": 20}, {\"price\": 30}]\\n"
                    "    assert calculate_total(items) == 60\\n"
                ),
                "utils.py": "def format_currency(value):\\n    return f'${value:.2f}'\\n",
            },
            file_tree=["README.md", "cart.py", "tests/test_cart.py", "utils.py"],
            failing_test="test_calculate_total_counts_all_items",
            stderr=(
                "FAILED tests/test_cart.py::test_calculate_total_counts_all_items\\n"
                "E assert 30 == 60\\n"
                "E where 30 = calculate_total([{'price': 10}, {'price': 20}, {'price': 30}])"
            ),
            bug_file="cart.py",
            bug_line=3,
            bug_summary="Off-by-one loop ignores final cart item.",
            symbols={
                "calculate_total": "cart.py",
                "test_calculate_total_counts_all_items": "tests/test_cart.py",
            },
            callers={
                "calculate_total": "tests/test_cart.py calls calculate_total from cart.py",
                "test_calculate_total_counts_all_items": "tests/test_cart.py -> cart.calculate_total",
            },
        ),
        BugCase(
            name="wrong_equality_auth",
            files={
                "README.md": "# Auth Utils\\nTiny login helper.\\n",
                "auth.py": (
                    "def is_admin(user):\\n"
                    "    if user.get(\"role\") != \"admin\":\\n"
                    "        return True\\n"
                    "    return False\\n"
                ),
                "tests/test_auth.py": (
                    "from auth import is_admin\\n\\n"
                    "def test_admin_user_is_admin():\\n"
                    "    assert is_admin({\"role\": \"admin\"}) is True\\n"
                ),
                "profile.py": "def display_name(user):\\n    return user.get('name', 'unknown')\\n",
            },
            file_tree=["README.md", "auth.py", "tests/test_auth.py", "profile.py"],
            failing_test="test_admin_user_is_admin",
            stderr=(
                "FAILED tests/test_auth.py::test_admin_user_is_admin\\n"
                "E assert False is True\\n"
                "E where False = is_admin({'role': 'admin'})"
            ),
            bug_file="auth.py",
            bug_line=2,
            bug_summary="Admin condition is inverted.",
            symbols={
                "is_admin": "auth.py",
                "test_admin_user_is_admin": "tests/test_auth.py",
            },
            callers={
                "is_admin": "tests/test_auth.py calls is_admin from auth.py",
                "test_admin_user_is_admin": "tests/test_auth.py -> auth.is_admin",
            },
        ),
        BugCase(
            name="unsafe_divide_guard_deleted",
            files={
                "README.md": "# Math Tools\\nBasic math helper functions.\\n",
                "math_tools.py": (
                    "def safe_divide(a, b):\\n"
                    "    return a / b\\n"
                ),
                "tests/test_math_tools.py": (
                    "from math_tools import safe_divide\\n\\n"
                    "def test_safe_divide_zero_returns_none():\\n"
                    "    assert safe_divide(10, 0) is None\\n"
                ),
                "stats.py": "def mean(values):\\n    return sum(values) / len(values)\\n",
            },
            file_tree=["README.md", "math_tools.py", "tests/test_math_tools.py", "stats.py"],
            failing_test="test_safe_divide_zero_returns_none",
            stderr=(
                "FAILED tests/test_math_tools.py::test_safe_divide_zero_returns_none\\n"
                "E ZeroDivisionError: division by zero\\n"
                "E where error occurred in safe_divide(10, 0)"
            ),
            bug_file="math_tools.py",
            bug_line=2,
            bug_summary="Missing zero division guard.",
            symbols={
                "safe_divide": "math_tools.py",
                "test_safe_divide_zero_returns_none": "tests/test_math_tools.py",
            },
            callers={
                "safe_divide": "tests/test_math_tools.py calls safe_divide from math_tools.py",
                "test_safe_divide_zero_returns_none": "tests/test_math_tools.py -> math_tools.safe_divide",
            },
        ),
        BugCase(
            name="wrong_variable_user_formatter",
            files={
                "README.md": "# User Formatting\\nHelpers for user display fields.\\n",
                "user_format.py": (
                    "def normalize_name(user):\\n"
                    "    username = user.get(\"username\", \"\")\\n"
                    "    display_name = user.get(\"display_name\", \"\")\\n"
                    "    return username.strip().title()\\n"
                ),
                "tests/test_user_format.py": (
                    "from user_format import normalize_name\\n\\n"
                    "def test_normalize_name_uses_display_name():\\n"
                    "    user = {\"username\": \"hidden_id\", \"display_name\": \"jane doe\"}\\n"
                    "    assert normalize_name(user) == \"Jane Doe\"\\n"
                ),
                "audit.py": "def log_event(name):\\n    return f'event={name}'\\n",
            },
            file_tree=["README.md", "user_format.py", "tests/test_user_format.py", "audit.py"],
            failing_test="test_normalize_name_uses_display_name",
            stderr=(
                "FAILED tests/test_user_format.py::test_normalize_name_uses_display_name\\n"
                "E assert 'Hidden_Id' == 'Jane Doe'\\n"
                "E where 'Hidden_Id' = normalize_name({'username': 'hidden_id', 'display_name': 'jane doe'})"
            ),
            bug_file="user_format.py",
            bug_line=4,
            bug_summary="Wrong variable returned; should use display_name.",
            symbols={
                "normalize_name": "user_format.py",
                "test_normalize_name_uses_display_name": "tests/test_user_format.py",
            },
            callers={
                "normalize_name": "tests/test_user_format.py calls normalize_name from user_format.py",
                "test_normalize_name_uses_display_name": "tests/test_user_format.py -> user_format.normalize_name",
            },
        ),
    ]
