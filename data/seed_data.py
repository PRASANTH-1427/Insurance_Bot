from __future__ import annotations


def seed_sample_data() -> list[dict[str, str]]:
    return [
        {"name": "Jane Doe", "email": "jane@example.com", "status": "new"},
        {"name": "John Smith", "email": "john@example.com", "status": "qualified"},
    ]
