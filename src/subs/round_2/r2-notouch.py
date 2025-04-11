import json
from abc import abstractmethod
from collections import deque
from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
from typing import Any, TypeAlias, List, Dict
import numpy as np

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


# PRODUCTS
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    # PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"


# LOGGER
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, List[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(
        self, order_depths: Dict[Symbol, OrderDepth]
    ) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


# STRAT SKELETON
class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


## MM SKELETON
class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = (
            len(self.window) == self.window_size
            and sum(self.window) >= self.window_size / 2
            and self.window[-1]
        )
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data, maxlen=self.window_size)


# KELPSTRAT (MM AROUND FAIR VAL)
class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return round((popular_buy_price + popular_sell_price) / 2)


# RESINSTRAT (MM AROUND FIXED VAL)
class ResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10000


# INKSTRAT (MEAN REVERSION)
class InkStrategy(Strategy):
    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        window_size: int = 50,
        threshold: int = 2,
        trade_unit: int = 5,
    ) -> None:
        super().__init__(symbol, limit)
        self.price_window = deque(maxlen=window_size)
        self.threshold = threshold
        self.trade_unit = trade_unit

    def act(self, state: TradingState) -> None:
        if self.symbol not in state.order_depths:
            return
        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = round((best_bid + best_ask) / 2)
        self.price_window.append(mid_price)
        if len(self.price_window) < self.price_window.maxlen:
            return
        moving_average = sum(self.price_window) / len(self.price_window)
        deviation = mid_price - moving_average
        position = state.position.get(self.symbol, 0)
        if deviation > self.threshold and position > -self.limit:
            quantity = min(self.trade_unit, position + self.limit)
            if quantity > 0:
                self.sell(mid_price, quantity)
        elif deviation < -self.threshold and position < self.limit:
            quantity = min(self.trade_unit, self.limit - position)
            if quantity > 0:
                self.buy(mid_price, quantity)

    def save(self) -> JSON:
        return list(self.price_window)

    def load(self, data: JSON) -> None:
        if data is not None:
            self.price_window = deque(data, maxlen=self.price_window.maxlen)
        else:
            self.price_window = deque(maxlen=self.price_window.maxlen)


# SPREADSTRAT
class SpreadStrategy(Strategy):
    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        spread_std_window: int = 45,
        spread_sma_window: int = 1500,
        zscore_threshold: int = 7,
        target_position: int = 58,
    ) -> None:
        super().__init__(symbol, limit)
        self.spread_history = []
        self.spread_std_window = spread_std_window
        self.spread_sma_window = spread_sma_window
        self.zscore_threshold = zscore_threshold
        self.target_position = target_position
        self.prev_zscore = 0
        self.clear_flag = False
        self.curr_mean = None
        # WEIGHTS
        self.basket_weights = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}

    def get_swmid(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[Symbol, OrderDepth]
    ) -> OrderDepth:
        CROISSANTS_PER_BASKET = self.basket_weights["CROISSANTS"]
        JAMS_PER_BASKET = self.basket_weights["JAMS"]
        DJEMBES_PER_BASKET = self.basket_weights["DJEMBES"]

        synthetic_order_depth = OrderDepth()

        croissants_best_bid = (
            max(order_depths["CROISSANTS"].buy_orders.keys())
            if order_depths["CROISSANTS"].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths["CROISSANTS"].sell_orders.keys())
            if order_depths["CROISSANTS"].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths["JAMS"].buy_orders.keys())
            if order_depths["JAMS"].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths["JAMS"].sell_orders.keys())
            if order_depths["JAMS"].sell_orders
            else float("inf")
        )
        djembes_best_bid = (
            max(order_depths["DJEMBES"].buy_orders.keys())
            if order_depths["DJEMBES"].buy_orders
            else 0
        )
        djembes_best_ask = (
            min(order_depths["DJEMBES"].sell_orders.keys())
            if order_depths["DJEMBES"].sell_orders
            else float("inf")
        )

        implied_bid = (
            croissants_best_bid * CROISSANTS_PER_BASKET
            + jams_best_bid * JAMS_PER_BASKET
            + djembes_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
            croissants_best_ask * CROISSANTS_PER_BASKET
            + jams_best_ask * JAMS_PER_BASKET
            + djembes_best_ask * DJEMBES_PER_BASKET
        )

        if implied_bid > 0:
            croissants_bid_volume = (
                order_depths["CROISSANTS"].buy_orders[croissants_best_bid]
                // CROISSANTS_PER_BASKET
            )
            jams_bid_volume = (
                order_depths["JAMS"].buy_orders[jams_best_bid] // JAMS_PER_BASKET
            )
            djembes_bid_volume = (
                order_depths["DJEMBES"].buy_orders[djembes_best_bid]
                // DJEMBES_PER_BASKET
            )
            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume, djembes_bid_volume
            )
            synthetic_order_depth.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = -(
                order_depths["CROISSANTS"].sell_orders[croissants_best_ask]
                // CROISSANTS_PER_BASKET
            )
            jams_ask_volume = -(
                order_depths["JAMS"].sell_orders[jams_best_ask] // JAMS_PER_BASKET
            )
            djembes_ask_volume = -(
                order_depths["DJEMBES"].sell_orders[djembes_best_ask]
                // DJEMBES_PER_BASKET
            )
            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume, djembes_ask_volume
            )
            synthetic_order_depth.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_depth

    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[Symbol, OrderDepth]
    ) -> Dict[Symbol, List[Order]]:
        component_orders = {
            "CROISSANTS": [],
            "JAMS": [],
            "DJEMBES": [],
        }
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            if quantity > 0 and price >= best_ask:
                croissants_price = min(order_depths["CROISSANTS"].sell_orders.keys())
                jams_price = min(order_depths["JAMS"].sell_orders.keys())
                djembes_price = min(order_depths["DJEMBES"].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                croissants_price = max(order_depths["CROISSANTS"].buy_orders.keys())
                jams_price = max(order_depths["JAMS"].buy_orders.keys())
                djembes_price = max(order_depths["DJEMBES"].buy_orders.keys())
            else:
                continue
            croissants_order = Order(
                "CROISSANTS",
                croissants_price,
                quantity * self.basket_weights["CROISSANTS"],
            )
            jams_order = Order(
                "JAMS", jams_price, quantity * self.basket_weights["JAMS"]
            )
            djembes_order = Order(
                "DJEMBES", djembes_price, quantity * self.basket_weights["DJEMBES"]
            )
            component_orders["CROISSANTS"].append(croissants_order)
            component_orders["JAMS"].append(jams_order)
            component_orders["DJEMBES"].append(djembes_order)
        return component_orders

    def execute_spread_orders(
        self,
        order_depths: Dict[Symbol, OrderDepth],
        target_position: int,
        basket_position: int,
    ) -> Dict[Symbol, List[Order]]:
        if target_position == basket_position:
            return None
        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths["PICNIC_BASKET1"]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])
            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )
            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [Order("PICNIC_BASKET1", basket_ask_price, execute_volume)]
            synthetic_orders = [
                Order("SYNTHETIC", synthetic_bid_price, -execute_volume)
            ]
            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders["PICNIC_BASKET1"] = basket_orders
            return aggregate_orders
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])
            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )
            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            basket_orders = [Order("PICNIC_BASKET1", basket_bid_price, -execute_volume)]
            synthetic_orders = [Order("SYNTHETIC", synthetic_ask_price, execute_volume)]
            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders["PICNIC_BASKET1"] = basket_orders
            return aggregate_orders

    def act(self, state: TradingState) -> None:
        # SPREAD: Synthetic - Basket
        if "PICNIC_BASKET1" not in state.order_depths:
            return
        basket_order_depth = state.order_depths["PICNIC_BASKET1"]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(
            state.order_depths
        )
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        self.spread_history.append(spread)
        if len(self.spread_history) < self.spread_std_window:
            return
        spread_std = np.std(self.spread_history[-self.spread_std_window :])
        if len(self.spread_history) == self.spread_sma_window:
            spread_mean = np.mean(self.spread_history)
            self.curr_mean = spread_mean
        elif len(self.spread_history) > self.spread_sma_window:
            spread_mean = self.curr_mean + (
                (spread - self.spread_history[0]) / self.spread_sma_window
            )
            self.spread_history.pop(0)
        else:
            spread_mean = self.params.get("default_spread_mean", 0)
        zscore = (spread - spread_mean) / spread_std
        position = state.position.get("PICNIC_BASKET1", 0)
        if zscore >= self.zscore_threshold:
            if position != -self.target_position:
                orders = self.execute_spread_orders(
                    state.order_depths, -self.target_position, position
                )
                if orders:
                    self.orders = []
                    for sym, orders_list in orders.items():
                        self.orders.extend(orders_list)
        elif zscore <= -self.zscore_threshold:
            if position != self.target_position:
                orders = self.execute_spread_orders(
                    state.order_depths, self.target_position, position
                )
                if orders:
                    self.orders = []
                    for sym, orders_list in orders.items():
                        self.orders.extend(orders_list)
        self.prev_zscore = zscore

    def save(self) -> JSON:
        return self.spread_history

    def load(self, data: JSON) -> None:
        self.spread_history = data if data is not None else []


# TRADER
class Trader:
    def __init__(self) -> None:
        limits = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 0,
            "PICNIC_BASKET1": 60,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
        }
        self.strategies: Dict[Symbol, Strategy] = {
            "KELP": KelpStrategy("KELP", limits["KELP"]),
            "RAINFOREST_RESIN": ResinStrategy(
                "RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]
            ),
            "SQUID_INK": InkStrategy("SQUID_INK", limits["SQUID_INK"]),
            "PICNIC_BASKET1": SpreadStrategy(
                "PICNIC_BASKET1",
                limits["PICNIC_BASKET1"],
                spread_std_window=45,
                spread_sma_window=1500,
                zscore_threshold=7,
                target_position=58,
            ),
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        logger.print(state.position)
        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
        orders: dict[Symbol, List[Order]] = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))
            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)
            new_trader_data[symbol] = strategy.save()
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
