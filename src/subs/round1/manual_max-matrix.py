import itertools
from typing import List, Tuple

# exchange matrix
prices = {
    "Snowball": {"Snowball": 1, "Pizza": 1.45, "Nuggets": 0.52, "Shells": 0.72},
    "Pizza": {"Snowball": 0.7, "Pizza": 1, "Nuggets": 0.31, "Shells": 0.48},
    "Nuggets": {"Snowball": 1.95, "Pizza": 3.1, "Nuggets": 1, "Shells": 1.49},
    "Shells": {"Snowball": 1.34, "Pizza": 1.98, "Nuggets": 0.64, "Shells": 1},
}

goods = list(prices.keys())


def calc_trade_value(path: List[str], start_amount: float) -> float:
    """simulate a series of trades and return the final amount."""
    amount = start_amount
    for i in range(len(path) - 1):
        amount *= prices[path[i]][path[i + 1]]
    return amount


def find_best_trade_path(
    start_good: str, start_amount: float, max_depth: int = 4
) -> Tuple[float, List[str]]:
    """find the most profitable path starting and ending with Seashells."""
    best_amount = start_amount
    best_path = [start_good]

    for depth in range(1, max_depth + 1):
        for middle_path in itertools.product(goods, repeat=depth):
            full_path = [start_good] + list(middle_path) + [start_good]
            if full_path == [start_good, start_good]:
                continue  # skip no-op trades

            final_amount = calc_trade_value(full_path, start_amount)
            if (final_amount > best_amount) or (
                final_amount == best_amount and len(full_path) < len(best_path)
            ):
                best_amount = final_amount
                best_path = full_path

    return best_amount, best_path


if __name__ == "__main__":
    start_good = "Shells"
    start_amount = 500

    best_amount, best_trade_path = find_best_trade_path(start_good, start_amount)

    print(f"Final amount: {best_amount:.2f}")
    print(f"Best trade path: {' -> '.join(best_trade_path)}")
