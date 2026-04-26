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
                "README.md": "# Cart Utils\nSmall checkout helper library.\n",
                "cart.py": (
                    "def calculate_total(items):\n"
                    "    total = 0\n"
                    "    for i in range(len(items) - 1):\n"
                    "        total += items[i][\"price\"]\n"
                    "    return total\n"
                ),
                "tests/test_cart.py": (
                    "from cart import calculate_total\n\n"
                    "def test_calculate_total_counts_all_items():\n"
                    "    items = [{\"price\": 10}, {\"price\": 20}, {\"price\": 30}]\n"
                    "    assert calculate_total(items) == 60\n"
                ),
                "utils.py": "def format_currency(value):\n    return f'${value:.2f}'\n",
            },
            file_tree=["README.md", "cart.py", "tests/test_cart.py", "utils.py"],
            failing_test="test_calculate_total_counts_all_items",
            stderr=(
                "FAILED tests/test_cart.py::test_calculate_total_counts_all_items\n"
                "E assert 30 == 60\n"
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
                "README.md": "# Auth Utils\nTiny login helper.\n",
                "auth.py": (
                    "def is_admin(user):\n"
                    "    if user.get(\"role\") != \"admin\":\n"
                    "        return True\n"
                    "    return False\n"
                ),
                "tests/test_auth.py": (
                    "from auth import is_admin\n\n"
                    "def test_admin_user_is_admin():\n"
                    "    assert is_admin({\"role\": \"admin\"}) is True\n"
                ),
                "profile.py": "def display_name(user):\n    return user.get('name', 'unknown')\n",
            },
            file_tree=["README.md", "auth.py", "tests/test_auth.py", "profile.py"],
            failing_test="test_admin_user_is_admin",
            stderr=(
                "FAILED tests/test_auth.py::test_admin_user_is_admin\n"
                "E assert False is True\n"
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
                "README.md": "# Math Tools\nBasic math helper functions.\n",
                "math_tools.py": (
                    "def safe_divide(a, b):\n"
                    "    return a / b\n"
                ),
                "tests/test_math_tools.py": (
                    "from math_tools import safe_divide\n\n"
                    "def test_safe_divide_zero_returns_none():\n"
                    "    assert safe_divide(10, 0) is None\n"
                ),
                "stats.py": "def mean(values):\n    return sum(values) / len(values)\n",
            },
            file_tree=["README.md", "math_tools.py", "tests/test_math_tools.py", "stats.py"],
            failing_test="test_safe_divide_zero_returns_none",
            stderr=(
                "FAILED tests/test_math_tools.py::test_safe_divide_zero_returns_none\n"
                "E ZeroDivisionError: division by zero\n"
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
                "README.md": "# User Formatting\nHelpers for user display fields.\n",
                "user_format.py": (
                    "def normalize_name(user):\n"
                    "    username = user.get(\"username\", \"\")\n"
                    "    display_name = user.get(\"display_name\", \"\")\n"
                    "    return username.strip().title()\n"
                ),
                "tests/test_user_format.py": (
                    "from user_format import normalize_name\n\n"
                    "def test_normalize_name_uses_display_name():\n"
                    "    user = {\"username\": \"hidden_id\", \"display_name\": \"jane doe\"}\n"
                    "    assert normalize_name(user) == \"Jane Doe\"\n"
                ),
                "audit.py": "def log_event(name):\n    return f'event={name}'\n",
            },
            file_tree=["README.md", "user_format.py", "tests/test_user_format.py", "audit.py"],
            failing_test="test_normalize_name_uses_display_name",
            stderr=(
                "FAILED tests/test_user_format.py::test_normalize_name_uses_display_name\n"
                "E assert 'Hidden_Id' == 'Jane Doe'\n"
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
        # --- NEW CASES BELOW ---
        BugCase(
            name="missing_return_value",
            files={
                "README.md": "# Order processing utils.\n",
                "orders.py": (
                    "def get_discount(order):\n"
                    "    if order.get(\"coupon\") == \"SAVE10\":\n"
                    "        discount = 10\n"
                    "    return 0\n"
                ),
                "tests/test_orders.py": (
                    "from orders import get_discount\n\n"
                    "def test_coupon_gives_discount():\n"
                    "    assert get_discount({\"coupon\": \"SAVE10\"}) == 10\n"
                ),
                "shipping.py": "def calc_shipping(weight):\n    return weight * 1.5\n",
            },
            file_tree=["README.md", "orders.py", "tests/test_orders.py", "shipping.py"],
            failing_test="test_coupon_gives_discount",
            stderr=(
                "FAILED tests/test_orders.py::test_coupon_gives_discount\n"
                "E assert 0 == 10\n"
                "E where 0 = get_discount({'coupon': 'SAVE10'})"
            ),
            bug_file="orders.py",
            bug_line=3,
            bug_summary="Missing return inside if-block; always falls through to return 0.",
            symbols={
                "get_discount": "orders.py",
                "test_coupon_gives_discount": "tests/test_orders.py",
            },
            callers={
                "get_discount": "tests/test_orders.py calls get_discount from orders.py",
                "test_coupon_gives_discount": "tests/test_orders.py -> orders.get_discount",
            },
        ),
        BugCase(
            name="list_mutation_side_effect",
            files={
                "README.md": "# Inventory helper.\n",
                "inventory.py": (
                    "def remove_item(stock, item):\n"
                    "    stock.remove(item)\n"
                    "    return stock\n"
                ),
                "tests/test_inventory.py": (
                    "from inventory import remove_item\n\n"
                    "def test_remove_does_not_mutate_original():\n"
                    "    original = [\"apple\", \"banana\", \"cherry\"]\n"
                    "    result = remove_item(original, \"banana\")\n"
                    "    assert original == [\"apple\", \"banana\", \"cherry\"]\n"
                ),
                "warehouse.py": "def restock(items, qty):\n    return items * qty\n",
            },
            file_tree=["README.md", "inventory.py", "tests/test_inventory.py", "warehouse.py"],
            failing_test="test_remove_does_not_mutate_original",
            stderr=(
                "FAILED tests/test_inventory.py::test_remove_does_not_mutate_original\n"
                "E assert ['apple', 'cherry'] == ['apple', 'banana', 'cherry']\n"
                "E where ['apple', 'cherry'] = original after remove_item call"
            ),
            bug_file="inventory.py",
            bug_line=2,
            bug_summary="Mutates the input list in-place instead of working on a copy.",
            symbols={
                "remove_item": "inventory.py",
                "test_remove_does_not_mutate_original": "tests/test_inventory.py",
            },
            callers={
                "remove_item": "tests/test_inventory.py calls remove_item from inventory.py",
                "test_remove_does_not_mutate_original": "tests/test_inventory.py -> inventory.remove_item",
            },
        ),
        BugCase(
            name="string_format_wrong_key",
            files={
                "README.md": "# Notification templates.\n",
                "notifier.py": (
                    "def build_message(user):\n"
                    "    return \"Hello, {name}! Your order is ready.\".format(\n"
                    "        username=user[\"name\"]\n"
                    "    )\n"
                ),
                "tests/test_notifier.py": (
                    "from notifier import build_message\n\n"
                    "def test_build_message_includes_name():\n"
                    "    msg = build_message({\"name\": \"Alice\"})\n"
                    "    assert \"Alice\" in msg\n"
                ),
                "email.py": "def send(to, body):\n    print(f'To: {to}\\n{body}')\n",
            },
            file_tree=["README.md", "notifier.py", "tests/test_notifier.py", "email.py"],
            failing_test="test_build_message_includes_name",
            stderr=(
                "FAILED tests/test_notifier.py::test_build_message_includes_name\n"
                "E KeyError: 'name'\n"
                "E where error occurred in build_message({'name': 'Alice'})"
            ),
            bug_file="notifier.py",
            bug_line=3,
            bug_summary="Format key is 'username' but template expects 'name'.",
            symbols={
                "build_message": "notifier.py",
                "test_build_message_includes_name": "tests/test_notifier.py",
            },
            callers={
                "build_message": "tests/test_notifier.py calls build_message from notifier.py",
                "test_build_message_includes_name": "tests/test_notifier.py -> notifier.build_message",
            },
        ),
        BugCase(
            name="wrong_comparison_type",
            files={
                "README.md": "# Age verification helper.\n",
                "age_check.py": (
                    "def is_adult(user):\n"
                    "    age = user.get(\"age\", \"0\")\n"
                    "    return age >= 18\n"
                ),
                "tests/test_age_check.py": (
                    "from age_check import is_adult\n\n"
                    "def test_adult_user_passes():\n"
                    "    assert is_adult({\"age\": 25}) is True\n\n"
                    "def test_minor_user_fails():\n"
                    "    assert is_adult({\"age\": 15}) is False\n"
                ),
                "profile.py": "def get_tier(age):\n    return 'senior' if age > 60 else 'standard'\n",
            },
            file_tree=["README.md", "age_check.py", "tests/test_age_check.py", "profile.py"],
            failing_test="test_adult_user_passes",
            stderr=(
                "FAILED tests/test_age_check.py::test_adult_user_passes\n"
                "E TypeError: '>=' not supported between instances of 'str' and 'int'\n"
                "E where error in is_adult({'age': 25})"
            ),
            bug_file="age_check.py",
            bug_line=2,
            bug_summary="Default age is string '0' not int 0; causes TypeError on comparison.",
            symbols={
                "is_adult": "age_check.py",
                "test_adult_user_passes": "tests/test_age_check.py",
            },
            callers={
                "is_adult": "tests/test_age_check.py calls is_adult from age_check.py",
                "test_adult_user_passes": "tests/test_age_check.py -> age_check.is_adult",
            },
        ),
        BugCase(
            name="index_out_of_range",
            files={
                "README.md": "# Queue manager.\n",
                "queue_mgr.py": (
                    "def peek_last(items):\n"
                    "    return items[len(items)]\n"
                ),
                "tests/test_queue_mgr.py": (
                    "from queue_mgr import peek_last\n\n"
                    "def test_peek_last_returns_last_item():\n"
                    "    assert peek_last([10, 20, 30]) == 30\n"
                ),
                "buffer.py": "def drain(items):\n    return items[:]\n",
            },
            file_tree=["README.md", "queue_mgr.py", "tests/test_queue_mgr.py", "buffer.py"],
            failing_test="test_peek_last_returns_last_item",
            stderr=(
                "FAILED tests/test_queue_mgr.py::test_peek_last_returns_last_item\n"
                "E IndexError: list index out of range\n"
                "E where error in peek_last([10, 20, 30])"
            ),
            bug_file="queue_mgr.py",
            bug_line=2,
            bug_summary="Should be items[len(items) - 1] or items[-1]; off-by-one causes IndexError.",
            symbols={
                "peek_last": "queue_mgr.py",
                "test_peek_last_returns_last_item": "tests/test_queue_mgr.py",
            },
            callers={
                "peek_last": "tests/test_queue_mgr.py calls peek_last from queue_mgr.py",
                "test_peek_last_returns_last_item": "tests/test_queue_mgr.py -> queue_mgr.peek_last",
            },
        ),
        BugCase(
            name="none_check_missing",
            files={
                "README.md": "# Config loader.\n",
                "config.py": (
                    "def get_timeout(config):\n"
                    "    return config[\"settings\"][\"timeout\"]\n"
                ),
                "tests/test_config.py": (
                    "from config import get_timeout\n\n"
                    "def test_missing_settings_returns_default():\n"
                    "    assert get_timeout({}) == 30\n"
                ),
                "defaults.py": "DEFAULT_TIMEOUT = 30\n",
            },
            file_tree=["README.md", "config.py", "tests/test_config.py", "defaults.py"],
            failing_test="test_missing_settings_returns_default",
            stderr=(
                "FAILED tests/test_config.py::test_missing_settings_returns_default\n"
                "E KeyError: 'settings'\n"
                "E where error in get_timeout({})"
            ),
            bug_file="config.py",
            bug_line=2,
            bug_summary="No fallback when 'settings' key is absent; should use .get() with default.",
            symbols={
                "get_timeout": "config.py",
                "test_missing_settings_returns_default": "tests/test_config.py",
            },
            callers={
                "get_timeout": "tests/test_config.py calls get_timeout from config.py",
                "test_missing_settings_returns_default": "tests/test_config.py -> config.get_timeout",
            },
        ),
    ]
