"""
Safe read-only order tools.

These functions are the security boundary — the agent can only do
what these functions allow. No raw SQL, no other tables.
"""

from datetime import datetime, timedelta
import random

from smalltask import tool


# --- Fake data store (replace with real DB calls) ---

_ORDERS = [
    {"id": f"ORD-{i:04d}", "user_id": f"USR-{(i % 20):03d}",
     "total": round(random.uniform(10, 500), 2),
     "status": random.choice(["completed", "completed", "completed", "refunded", "pending"]),
     "created_at": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()}
    for i in range(1, 101)
]


@tool
def get_orders_last_n_days(days: int) -> list:
    """Return all orders placed in the last N days.

    Args:
        days: Number of days to look back.
    """
    cutoff = datetime.now() - timedelta(days=days)
    return [
        o for o in _ORDERS
        if datetime.fromisoformat(o["created_at"]) >= cutoff
    ]


@tool
def get_order_summary(days: int) -> dict:
    """Return aggregated order stats for the last N days.

    Args:
        days: Number of days to look back.
    """
    orders = get_orders_last_n_days(days)
    if not orders:
        return {"total_orders": 0, "total_revenue": 0.0, "refund_rate": 0.0}

    total_revenue = sum(o["total"] for o in orders if o["status"] == "completed")
    refunds = sum(1 for o in orders if o["status"] == "refunded")

    return {
        "total_orders": len(orders),
        "total_revenue": round(total_revenue, 2),
        "refund_count": refunds,
        "refund_rate": round(refunds / len(orders) * 100, 1),
        "avg_order_value": round(total_revenue / max(len(orders), 1), 2),
    }


@tool
def get_top_customers(days: int, limit: int) -> list:
    """Return the top customers by spend in the last N days.

    Args:
        days: Number of days to look back.
        limit: Maximum number of customers to return.
    """
    orders = get_orders_last_n_days(days)
    spend: dict[str, float] = {}
    for o in orders:
        if o["status"] == "completed":
            spend[o["user_id"]] = spend.get(o["user_id"], 0) + o["total"]

    sorted_customers = sorted(spend.items(), key=lambda x: x[1], reverse=True)
    return [
        {"user_id": uid, "total_spend": round(amount, 2)}
        for uid, amount in sorted_customers[:limit]
    ]
