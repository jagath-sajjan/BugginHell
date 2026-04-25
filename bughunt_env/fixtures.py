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
                "utils.py": (
                    "def format_currency(value):\\n"
                    "    return f'${value:.2f}'\\n"
                ),
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
            bug_summary="Off-by-one loop ignores the final cart item.",
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
                "profile.py": (
                    "def display_name(user):\\n"
                    "    return user.get('name', 'unknown')\\n"
                ),
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
    ]
