"""
Standard benchmark tasks for GUI Agent evaluation.

Each task represents a common GUI interaction scenario.
Inspired by real-world conditions in WebArena, Mind2Web, ScreenSpot, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from benchmark.runner import BenchmarkTask, BenchmarkTaskType


# ---------------------------------------------------------------------------
# Task Loader
# ---------------------------------------------------------------------------

class TaskLoader:
    """Load and manage benchmark tasks."""

    @staticmethod
    def load_standard_tasks() -> List[BenchmarkTask]:
        """Return the standard set of benchmark tasks."""
        tasks: List[BenchmarkTask] = []

        # ---- Click Tasks ----
        tasks.extend(_click_tasks())
        tasks.extend(_type_tasks())
        tasks.extend(_scroll_tasks())
        tasks.extend(_navigate_tasks())
        tasks.extend(_form_tasks())
        tasks.extend(_search_tasks())
        tasks.extend(_complex_tasks())

        return tasks

    @staticmethod
    def load_from_json(file_path: Path) -> List[BenchmarkTask]:
        """Load tasks from a JSON file."""
        import json

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        tasks = []
        for item in data:
            task = BenchmarkTask(
                id=item["id"],
                task_type=BenchmarkTaskType(item.get("task_type", "click")),
                description=item.get("description", ""),
                instruction=item.get("instruction", ""),
                screenshot_path=Path(item["screenshot_path"]) if item.get("screenshot_path") else None,
                expected_actions=item.get("expected_actions", []),
                max_steps=item.get("max_steps", 10),
                timeout_sec=item.get("timeout_sec", 30),
                metadata=item.get("metadata", {}),
            )
            tasks.append(task)

        return tasks


# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

def _click_tasks() -> List[BenchmarkTask]:
    return [
        BenchmarkTask(
            id="click_button_basic",
            task_type=BenchmarkTaskType.CLICK,
            description="Click a standard button",
            instruction='Click the "Submit" button',
            expected_actions=[
                {"action_type": "click", "target": "Submit", "type": "button"},
            ],
            max_steps=1,
            metadata={"difficulty": "easy", "tags": ["button", "web"]},
        ),
        BenchmarkTask(
            id="click_link_navigation",
            task_type=BenchmarkTaskType.CLICK,
            description="Click a navigation link",
            instruction='Click the "About Us" link in the header',
            expected_actions=[
                {"action_type": "click", "target": "About Us", "type": "link"},
            ],
            max_steps=1,
            metadata={"difficulty": "easy", "tags": ["link", "web", "navigation"]},
        ),
        BenchmarkTask(
            id="click_small_element",
            task_type=BenchmarkTaskType.CLICK,
            description="Click a small icon or button",
            instruction="Click the settings gear icon",
            expected_actions=[
                {"action_type": "click", "target": "settings_icon", "type": "icon"},
            ],
            max_steps=1,
            metadata={"difficulty": "medium", "tags": ["icon", "small"]},
        ),
        BenchmarkTask(
            id="click_dropdown_option",
            task_type=BenchmarkTaskType.CLICK,
            description="Select an option from a dropdown",
            instruction='Select "United States" from the country dropdown',
            expected_actions=[
                {"action_type": "click", "target": "dropdown", "type": "select"},
                {"action_type": "click", "target": "United States", "type": "option"},
            ],
            max_steps=2,
            metadata={"difficulty": "medium", "tags": ["dropdown", "select"]},
        ),
        BenchmarkTask(
            id="click_checkbox",
            task_type=BenchmarkTaskType.CLICK,
            description="Check a checkbox",
            instruction='Check the "I agree to the terms" checkbox',
            expected_actions=[
                {"action_type": "click", "target": "terms_checkbox", "type": "checkbox"},
            ],
            max_steps=1,
            metadata={"difficulty": "easy", "tags": ["checkbox", "form"]},
        ),
    ]


def _type_tasks() -> List[BenchmarkTask]:
    return [
        BenchmarkTask(
            id="type_text_basic",
            task_type=BenchmarkTaskType.TYPE,
            description="Type into a text field",
            instruction='Type "hello@example.com" into the email field',
            expected_actions=[
                {
                    "action_type": "type",
                    "target": "email_field",
                    "text": "hello@example.com",
                },
            ],
            max_steps=2,
            metadata={"difficulty": "easy", "tags": ["input", "text"]},
        ),
        BenchmarkTask(
            id="type_multiline",
            task_type=BenchmarkTaskType.TYPE,
            description="Type multi-line text into a textarea",
            instruction='Type "Dear customer,\nThank you for your inquiry.\nBest regards" into the message textarea',
            expected_actions=[
                {
                    "action_type": "type",
                    "target": "message_textarea",
                    "text": "Dear customer,\nThank you for your inquiry.\nBest regards",
                },
            ],
            max_steps=2,
            metadata={"difficulty": "medium", "tags": ["textarea", "multiline"]},
        ),
        BenchmarkTask(
            id="type_password",
            task_type=BenchmarkTaskType.TYPE,
            description="Type into a password field",
            instruction='Type "P@ssw0rd123" into the password field',
            expected_actions=[
                {
                    "action_type": "type",
                    "target": "password_field",
                    "text": "P@ssw0rd123",
                },
            ],
            max_steps=1,
            metadata={"difficulty": "easy", "tags": ["password", "input"]},
        ),
        BenchmarkTask(
            id="type_search",
            task_type=BenchmarkTaskType.TYPE,
            description="Type a search query",
            instruction='Type "artificial intelligence" into the search bar',
            expected_actions=[
                {
                    "action_type": "type",
                    "target": "search_bar",
                    "text": "artificial intelligence",
                },
            ],
            max_steps=2,
            metadata={"difficulty": "easy", "tags": ["search", "input"]},
        ),
    ]


def _scroll_tasks() -> List[BenchmarkTask]:
    return [
        BenchmarkTask(
            id="scroll_down_basic",
            task_type=BenchmarkTaskType.SCROLL,
            description="Scroll down the page",
            instruction="Scroll down to see more content",
            expected_actions=[
                {"action_type": "scroll", "direction": "down", "amount": 500},
            ],
            max_steps=1,
            metadata={"difficulty": "easy", "tags": ["scroll", "page"]},
        ),
        BenchmarkTask(
            id="scroll_to_element",
            task_type=BenchmarkTaskType.SCROLL,
            description="Scroll until a specific element is visible",
            instruction='Scroll down until you see the "Contact Us" section',
            expected_actions=[
                {"action_type": "scroll", "direction": "down"},
            ],
            max_steps=5,
            metadata={"difficulty": "medium", "tags": ["scroll", "element"]},
        ),
        BenchmarkTask(
            id="scroll_horizontal",
            task_type=BenchmarkTaskType.SCROLL,
            description="Scroll horizontally",
            instruction="Scroll right to see more table columns",
            expected_actions=[
                {"action_type": "scroll", "direction": "right", "amount": 300},
            ],
            max_steps=1,
            metadata={"difficulty": "medium", "tags": ["scroll", "table"]},
        ),
    ]


def _navigate_tasks() -> List[BenchmarkTask]:
    return [
        BenchmarkTask(
            id="navigate_to_url",
            task_type=BenchmarkTaskType.NAVIGATE,
            description="Navigate to a URL",
            instruction="Go to https://www.example.com",
            expected_actions=[
                {"action_type": "navigate", "url": "https://www.example.com"},
            ],
            max_steps=1,
            metadata={"difficulty": "easy", "tags": ["navigation", "url"]},
        ),
        BenchmarkTask(
            id="navigate_back",
            task_type=BenchmarkTaskType.NAVIGATE,
            description="Navigate back to previous page",
            instruction="Go back to the previous page",
            expected_actions=[
                {"action_type": "navigate", "direction": "back"},
            ],
            max_steps=1,
            metadata={"difficulty": "easy", "tags": ["navigation", "back"]},
        ),
        BenchmarkTask(
            id="navigate_refresh",
            task_type=BenchmarkTaskType.NAVIGATE,
            description="Refresh the current page",
            instruction="Refresh the page",
            expected_actions=[
                {"action_type": "navigate", "direction": "refresh"},
            ],
            max_steps=1,
            metadata={"difficulty": "easy", "tags": ["navigation", "refresh"]},
        ),
    ]


def _form_tasks() -> List[BenchmarkTask]:
    return [
        BenchmarkTask(
            id="form_login",
            task_type=BenchmarkTaskType.FORM,
            description="Complete a login form",
            instruction='Fill the login form: email "user@test.com", password "pass123"',
            expected_actions=[
                {"action_type": "type", "target": "email", "text": "user@test.com"},
                {"action_type": "type", "target": "password", "text": "pass123"},
                {"action_type": "click", "target": "login_button"},
            ],
            max_steps=4,
            metadata={"difficulty": "medium", "tags": ["form", "login"]},
        ),
        BenchmarkTask(
            id="form_registration",
            task_type=BenchmarkTaskType.FORM,
            description="Complete a multi-field registration form",
            instruction='Fill the registration form: name "Alice", email "alice@example.com", phone "1234567890", country "Canada"',
            expected_actions=[
                {"action_type": "type", "target": "name", "text": "Alice"},
                {"action_type": "type", "target": "email", "text": "alice@example.com"},
                {"action_type": "type", "target": "phone", "text": "1234567890"},
                {"action_type": "click", "target": "country_dropdown"},
                {"action_type": "click", "target": "Canada"},
                {"action_type": "click", "target": "submit"},
            ],
            max_steps=8,
            metadata={"difficulty": "hard", "tags": ["form", "registration"]},
        ),
        BenchmarkTask(
            id="form_with_validation",
            task_type=BenchmarkTaskType.FORM,
            description="Complete a form with field validation",
            instruction='Fill the form: zip code "90210", credit card "4111111111111111", expiry "12/28", CVV "123"',
            expected_actions=[
                {"action_type": "type", "target": "zip", "text": "90210"},
                {"action_type": "type", "target": "card", "text": "4111111111111111"},
                {"action_type": "type", "target": "expiry", "text": "12/28"},
                {"action_type": "type", "target": "cvv", "text": "123"},
                {"action_type": "click", "target": "pay_button"},
            ],
            max_steps=6,
            metadata={"difficulty": "hard", "tags": ["form", "payment"]},
        ),
    ]


def _search_tasks() -> List[BenchmarkTask]:
    return [
        BenchmarkTask(
            id="search_and_click_result",
            task_type=BenchmarkTaskType.SEARCH,
            description="Search and click the first result",
            instruction='Search for "latest news" and click the first result',
            expected_actions=[
                {"action_type": "type", "target": "search", "text": "latest news"},
                {"action_type": "click", "target": "search_button"},
                {"action_type": "click", "target": "first_result"},
            ],
            max_steps=5,
            metadata={"difficulty": "medium", "tags": ["search", "results"]},
        ),
        BenchmarkTask(
            id="search_filtered",
            task_type=BenchmarkTaskType.SEARCH,
            description="Search with filters",
            instruction='Search for "laptops", then apply filter: price under $1000',
            expected_actions=[
                {"action_type": "type", "target": "search", "text": "laptops"},
                {"action_type": "click", "target": "search_button"},
                {"action_type": "click", "target": "price_filter"},
                {"action_type": "type", "target": "max_price", "text": "1000"},
            ],
            max_steps=6,
            metadata={"difficulty": "hard", "tags": ["search", "filter"]},
        ),
    ]


def _complex_tasks() -> List[BenchmarkTask]:
    return [
        BenchmarkTask(
            id="complex_book_flight",
            task_type=BenchmarkTaskType.COMPLEX,
            description="Book a flight: search, select, fill passenger info",
            instruction='Search for a flight from New York to Los Angeles on July 15, pick the first result, and fill passenger details: name "John Doe", email "john@example.com"',
            expected_actions=[
                {"action_type": "type", "target": "origin", "text": "New York"},
                {"action_type": "type", "target": "destination", "text": "Los Angeles"},
                {"action_type": "click", "target": "date_picker"},
                {"action_type": "click", "target": "july_15"},
                {"action_type": "click", "target": "search_flights"},
                {"action_type": "click", "target": "first_flight"},
                {"action_type": "type", "target": "passenger_name", "text": "John Doe"},
                {"action_type": "type", "target": "passenger_email", "text": "john@example.com"},
            ],
            max_steps=12,
            timeout_sec=60,
            metadata={"difficulty": "extreme", "tags": ["multi-step", "travel"]},
        ),
        BenchmarkTask(
            id="complex_shopping_cart",
            task_type=BenchmarkTaskType.COMPLEX,
            description="Add item to cart and checkout",
            instruction='Find "Wireless Mouse", add to cart, view cart, and proceed to checkout',
            expected_actions=[
                {"action_type": "type", "target": "search", "text": "Wireless Mouse"},
                {"action_type": "click", "target": "search_button"},
                {"action_type": "click", "target": "product_result"},
                {"action_type": "click", "target": "add_to_cart"},
                {"action_type": "click", "target": "view_cart"},
                {"action_type": "click", "target": "checkout"},
            ],
            max_steps=10,
            timeout_sec=60,
            metadata={"difficulty": "hard", "tags": ["e-commerce", "multi-step"]},
        ),
        BenchmarkTask(
            id="complex_tabular_data",
            task_type=BenchmarkTaskType.COMPLEX,
            description="Interact with a data table: sort, filter, extract",
            instruction='Sort the table by "Price" column descending, then click the row with the highest price',
            expected_actions=[
                {"action_type": "click", "target": "price_column_header"},
                {"action_type": "click", "target": "price_column_header"},
                {"action_type": "click", "target": "first_row"},
            ],
            max_steps=5,
            timeout_sec=30,
            metadata={"difficulty": "medium", "tags": ["table", "data"]},
        ),
        BenchmarkTask(
            id="complex_modal_dialog",
            task_type=BenchmarkTaskType.COMPLEX,
            description="Handle a modal dialog",
            instruction='Click "Delete Account", then confirm in the modal dialog',
            expected_actions=[
                {"action_type": "click", "target": "delete_account_button"},
                {"action_type": "click", "target": "confirm_delete_modal"},
            ],
            max_steps=3,
            metadata={"difficulty": "medium", "tags": ["modal", "dialog"]},
        ),
    ]


# ---------------------------------------------------------------------------
# Task difficulty distribution
# ---------------------------------------------------------------------------

def task_difficulty_summary(tasks: List[BenchmarkTask]) -> Dict[str, int]:
    """Get task count by difficulty."""
    summary: Dict[str, int] = {}
    for task in tasks:
        diff = task.metadata.get("difficulty", "unknown")
        summary[diff] = summary.get(diff, 0) + 1
    return summary


def task_type_summary(tasks: List[BenchmarkTask]) -> Dict[str, int]:
    """Get task count by type."""
    summary: Dict[str, int] = {}
    for task in tasks:
        summary[task.task_type.value] = summary.get(task.task_type.value, 0) + 1
    return summary
