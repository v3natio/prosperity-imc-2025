import itertools
import json
import numpy as np
from abc import abstractmethod
from datamodel import Order, OrderDepth, Symbol, TradingState
from enum import IntEnum
from pathlib import Path
from prosperity3bt.data import read_day_data
from prosperity3bt.file_reader import PackageResourcesReader
from prosperity3bt.models import (
    ActivityLogRow,
    BacktestResult,
    MarketTrade,
    SandboxLogRow,
    TradeMatchingMode,
    TradeRow,
)
from prosperity3bt.runner import run_backtest
from tqdm.contrib.concurrent import process_map
from typing import TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0

        self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class Signal(IntEnum):
    NEUTRAL = 0
    SHORT = 1
    LONG = 2


class SignalStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.signal = Signal.NEUTRAL

    @abstractmethod
    def get_signal(self, state: TradingState) -> Signal | None:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        new_signal = self.get_signal(state)
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        if self.signal == Signal.NEUTRAL:
            if position < 0:
                self.buy(self.get_buy_price(order_depth), -position)
            elif position > 0:
                self.sell(self.get_sell_price(order_depth), position)
        elif self.signal == Signal.SHORT:
            self.sell(self.get_sell_price(order_depth), self.limit + position)
        elif self.signal == Signal.LONG:
            self.buy(self.get_buy_price(order_depth), self.limit - position)

    def get_buy_price(self, order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders.keys())

    def get_sell_price(self, order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders.keys())

    def save(self) -> JSON:
        return self.signal.value

    def load(self, data: JSON) -> None:
        self.signal = Signal(data)


class MyStrategy(SignalStrategy):
    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        buyer1: str,
        seller1: str,
        buyer2: str,
        seller2: str,
    ) -> None:
        super().__init__(symbol, limit)

        self.buyer1 = buyer1
        self.seller1 = seller1
        self.buyer2 = buyer2
        self.seller2 = seller2

    def get_signal(self, state: TradingState) -> Signal | None:
        trades = state.market_trades.get(self.symbol, [])
        trades = [t for t in trades if t.timestamp == state.timestamp - 100]

        if any(t.buyer == self.buyer1 and t.seller == self.seller1 for t in trades):
            return Signal.LONG

        if any(t.buyer == self.buyer2 and t.seller == self.seller2 for t in trades):
            return Signal.SHORT


class Trader:
    def __init__(self, buyer1: str, seller1: str, buyer2: str, seller2: str) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75,
        }

        self.strategies = {
            symbol: MyStrategy(symbol, limit, buyer1, seller1, buyer2, seller2)
            for symbol, limit in limits.items()
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0
        trader_data = ""

        for symbol, strategy in self.strategies.items():
            if (
                symbol in state.order_depths
                and len(state.order_depths[symbol].buy_orders) > 0
                and len(state.order_depths[symbol].sell_orders) > 0
            ):
                strategy_orders, strategy_conversions = strategy.run(state)
                orders[symbol] = strategy_orders
                conversions += strategy_conversions

        return orders, conversions, trader_data


def run(buyer1: str, seller1: str, buyer2: str, seller2: str) -> dict[str, float]:
    file_reader = PackageResourcesReader()

    out = {
        "buyer1": buyer1,
        "seller1": seller1,
        "buyer2": buyer2,
        "seller2": seller2,
        "total_pnl": 0,
    }

    products = list(
        {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75,
        }.keys()
    )

    for product in products:
        out[f"total_{product}_pnl"] = 0
        out[f"{product}_min"] = 1e9
        out[f"{product}_max"] = -1e9

    days = [[5, [2, 3, 4]]]

    for round_num, day_nums in days:
        for day_num in day_nums:
            trader = Trader(buyer1, seller1, buyer2, seller2)
            result = run_backtest(
                trader,
                file_reader,
                round_num,
                day_num,
                print_output=False,
                trade_matching_mode=TradeMatchingMode.all,
                no_names=False,
                show_progress_bar=True,
            )

            out[f"round{round_num}_day{day_num}_pnl"] = 0

            for product in products:
                pnls = [
                    row.columns[-1]
                    for row in result.activity_logs
                    if row.columns[2] == product
                ]

                min_pnl = min(pnls) if len(pnls) > 0 else 0
                max_pnl = max(pnls) if len(pnls) > 0 else 0
                final_pnl = pnls[-1] if len(pnls) > 0 else 0

                out[f"round{round_num}_day{day_num}_pnl"] += final_pnl
                out[f"round{round_num}_day{day_num}_{product}_pnl"] = final_pnl

                out["total_pnl"] += final_pnl
                out[f"total_{product}_pnl"] += final_pnl

                out[f"round{round_num}_day{day_num}_{product}_min"] = min_pnl
                out[f"round{round_num}_day{day_num}_{product}_max"] = max_pnl

                out[f"{product}_min"] = min(out[f"{product}_min"], min_pnl)
                out[f"{product}_max"] = max(out[f"{product}_max"], max_pnl)

    return out


combinations = [
    ("Caesar", "Caesar"),
    ("Caesar", "Camilla"),
    ("Caesar", "Gary"),
    ("Caesar", "Gina"),
    ("Caesar", "Olivia"),
    ("Caesar", "Pablo"),
    ("Caesar", "Paris"),
    ("Caesar", "Penelope"),
    ("Caesar", "Peter"),
    ("Camilla", "Caesar"),
    ("Camilla", "Camilla"),
    ("Camilla", "Gary"),
    ("Camilla", "Gina"),
    ("Camilla", "Pablo"),
    ("Camilla", "Paris"),
    ("Camilla", "Penelope"),
    ("Charlie", "Caesar"),
    ("Charlie", "Camilla"),
    ("Charlie", "Gary"),
    ("Charlie", "Gina"),
    ("Charlie", "Olivia"),
    ("Charlie", "Pablo"),
    ("Charlie", "Paris"),
    ("Charlie", "Penelope"),
    ("Gary", "Caesar"),
    ("Gary", "Camilla"),
    ("Gary", "Charlie"),
    ("Gary", "Gina"),
    ("Gary", "Olga"),
    ("Gary", "Pablo"),
    ("Gary", "Paris"),
    ("Gary", "Penelope"),
    ("Gina", "Caesar"),
    ("Gina", "Camilla"),
    ("Gina", "Charlie"),
    ("Gina", "Gary"),
    ("Gina", "Pablo"),
    ("Gina", "Paris"),
    ("Gina", "Penelope"),
    ("Olivia", "Caesar"),
    ("Olivia", "Charlie"),
    ("Pablo", "Caesar"),
    ("Pablo", "Camilla"),
    ("Pablo", "Charlie"),
    ("Pablo", "Gary"),
    ("Pablo", "Gina"),
    ("Pablo", "Pablo"),
    ("Pablo", "Paris"),
    ("Pablo", "Penelope"),
    ("Paris", "Caesar"),
    ("Paris", "Camilla"),
    ("Paris", "Charlie"),
    ("Paris", "Gary"),
    ("Paris", "Gina"),
    ("Paris", "Olivia"),
    ("Paris", "Pablo"),
    ("Paris", "Paris"),
    ("Paris", "Penelope"),
    ("Penelope", "Caesar"),
    ("Penelope", "Camilla"),
    ("Penelope", "Charlie"),
    ("Penelope", "Gina"),
    ("Penelope", "Pablo"),
    ("Penelope", "Paris"),
    ("Peter", "Caesar"),
]

buyer1_values = []
seller1_values = []
buyer2_values = []
seller2_values = []

for buyer1, seller1 in combinations:
    for buyer2, seller2 in combinations:
        buyer1_values.append(buyer1)
        seller1_values.append(seller1)
        buyer2_values.append(buyer2)
        seller2_values.append(seller2)

results = process_map(
    run,
    buyer1_values,
    seller1_values,
    buyer2_values,
    seller2_values,
    max_workers=12,
    chunksize=1,
    ascii=True,
)

output_file = Path(__file__).parent / f"{Path(__file__).stem}.json"
with output_file.open("w+", encoding="utf-8") as file:
    file.write(json.dumps(results, separators=(",", ":")))
