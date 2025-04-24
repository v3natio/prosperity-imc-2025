import json
import math
import statistics
import numpy as np
from statistics import NormalDist
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
from typing import Any, TypeAlias, Dict

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
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

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


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

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
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
        self.window = deque(data)


class SignalStrategy(Strategy):
    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.sell_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position

        self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.buy_orders.keys())

        position = state.position.get(self.symbol, 0)
        to_sell = self.limit + position

        self.sell(price, to_sell)


class BasketStrategy(Strategy):
    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        basket_weights: dict,
        default_spread_mean: float,
        default_spread_std: float,
        spread_std_window: int,
        zscore_threshold: float,
        target_position: int,
    ) -> None:
        super().__init__(symbol, limit)

        # params
        self.default_spread_mean = default_spread_mean
        self.default_spread_std = default_spread_std
        self.spread_std_window = spread_std_window
        self.zscore_threshold = zscore_threshold
        self.target_position = target_position

        # weights
        self.basket_weights = basket_weights

        # start spread hist
        self.spread_history = deque(maxlen=self.spread_std_window)
        self.prev_zscore = 0

    def get_swmid(self, order_depth) -> float:
        # calc volume weighted mid px
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(self, state: TradingState) -> OrderDepth:
        order_depths = state.order_depths
        basket_composition = self.basket_weights

        synthetic_order_price = OrderDepth()

        # track best bid/ask of constituents
        component_best_bids = {}
        component_best_asks = {}

        # calc best bid/ask of constituents
        for component, weight in basket_composition.items():
            if component in order_depths and order_depths[component].buy_orders:
                component_best_bids[component] = max(
                    order_depths[component].buy_orders.keys()
                )
            else:
                component_best_bids[component] = 0

            if component in order_depths and order_depths[component].sell_orders:
                component_best_asks[component] = min(
                    order_depths[component].sell_orders.keys()
                )
            else:
                component_best_asks[component] = float("inf")

        # calc implied bid/ask of synthetic basket
        implied_bid = sum(
            component_best_bids[component] * weight
            for component, weight in basket_composition.items()
        )
        implied_ask = sum(
            component_best_asks[component] * weight
            for component, weight in basket_composition.items()
        )

        # calculate max synthetic baskets possible based on implied bid/ask
        if implied_bid > 0:
            # calculate synthetic basket amount based on constituent volume
            implied_bid_volumes = []
            for component, weight in basket_composition.items():
                if component_best_bids[component] > 0:
                    component_volume = order_depths[component].buy_orders[
                        component_best_bids[component]
                    ]
                    implied_bid_volumes.append(component_volume // weight)

            if implied_bid_volumes:  # constituent volume check
                implied_bid_volume = min(implied_bid_volumes)
                synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            # calculate synthetic basket amount based on constituent volume
            implied_ask_volumes = []
            for component, weight in basket_composition.items():
                if component_best_asks[component] < float("inf"):
                    component_volume = -order_depths[component].sell_orders[
                        component_best_asks[component]
                    ]
                    implied_ask_volumes.append(component_volume // weight)

            if implied_ask_volumes:  # product volume check
                implied_ask_volume = min(implied_ask_volumes)
                synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def act(self, state: TradingState) -> None:
        # check all products are in order depth
        basket_type = self.symbol
        if any(
            symbol not in state.order_depths
            for symbol in list(self.basket_weights.keys()) + [basket_type]
        ):
            return

        basket_order_depth = state.order_depths[basket_type]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(state)

        # check order depth
        if (
            not basket_order_depth.buy_orders
            or not basket_order_depth.sell_orders
            or not synthetic_order_depth.buy_orders
            or not synthetic_order_depth.sell_orders
        ):
            return

        # calc spread
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid

        # update spread hist
        self.spread_history.append(spread)

        # enough data check
        if len(self.spread_history) < self.spread_std_window:
            return

        # calc zscore
        spread_std = np.std(list(self.spread_history))
        zscore = (spread - self.default_spread_mean) / spread_std

        # get curr pos
        position = state.position.get(basket_type, 0)

        # zscore threshold
        if zscore >= self.zscore_threshold:
            if position > -self.target_position:
                quantity_to_sell = min(
                    position + self.target_position, self.limit + position
                )
                if quantity_to_sell > 0:
                    # sell at best bid
                    price = max(basket_order_depth.buy_orders.keys())
                    self.sell(price, quantity_to_sell)

        elif zscore <= -self.zscore_threshold:
            if position < self.target_position:
                quantity_to_buy = min(
                    self.target_position - position, self.limit - position
                )
                if quantity_to_buy > 0:
                    # buy at best ask
                    price = min(basket_order_depth.sell_orders.keys())
                    self.buy(price, quantity_to_buy)

        self.prev_zscore = zscore

    def save(self) -> JSON:
        return {
            "spread_history": list(self.spread_history),
            "prev_zscore": self.prev_zscore,
        }

    def load(self, data: JSON) -> None:
        if data is not None:
            if "spread_history" in data:
                self.spread_history = deque(
                    data["spread_history"], maxlen=self.spread_std_window
                )
            if "prev_zscore" in data:
                self.prev_zscore = data["prev_zscore"]


def BS_CALL(S, K, T, r, sigma):
    N = NormalDist().cdf
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * N(d1) - K * math.exp(-r * T) * N(d2)


def BS_PUT(S, K, T, r, sigma):
    N = NormalDist().cdf
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


class BlackScholesStrategy(SignalStrategy):
    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        underlying_symbol: str,
        strike_price: int,
        initial_time_to_expiry: float,
        option_type: str = "call",
        history_length: int = 50,
        threshold: float = 2.0,
    ) -> None:
        super().__init__(symbol, limit)

        # option params
        self.underlying_symbol = underlying_symbol
        self.strike_price = strike_price
        self.initial_time_to_expiry = initial_time_to_expiry
        self.option_type = option_type.lower()  # call/put
        self.threshold = threshold

        # vol calc
        self.history_length = history_length
        self.price_history = deque(maxlen=history_length)

    def update_price_history(self, new_price: float) -> None:
        # update px hist from mid_price
        self.price_history.append(new_price)

    def compute_sigma(self) -> float:
        # realized vol from px hist
        if len(self.price_history) < 3:
            raise ValueError("Not enough data to compute sigma")

        log_returns = [
            math.log(self.price_history[i + 1] / self.price_history[i])
            for i in range(len(self.price_history) - 1)
        ]

        if len(log_returns) <= 1:
            raise ValueError("Not enough log returns to compute standard deviation")

        stdev = statistics.stdev(log_returns)
        sigma = stdev * math.sqrt(252)
        return sigma

    def act(self, state: TradingState) -> None:
        # Ensure both markets have viable order depths
        if (
            self.underlying_symbol not in state.order_depths
            or not state.order_depths[self.underlying_symbol].buy_orders
            or not state.order_depths[self.underlying_symbol].sell_orders
        ):
            return

        if (
            self.symbol not in state.order_depths
            or not state.order_depths[self.symbol].buy_orders
            or not state.order_depths[self.symbol].sell_orders
        ):
            return

        underlying_price = self.get_mid_price(state, self.underlying_symbol)
        option_price = self.get_mid_price(state, self.symbol)

        self.update_price_history(underlying_price)

        try:
            sigma = self.compute_sigma()
        except ValueError:
            return

        S = underlying_price
        K = self.strike_price
        r = 0

        mill = 1_000_000  # ticks per day
        frac = state.timestamp / mill  # how much of the curr day has passed
        T = self.initial_time_to_expiry - (frac / 365)

        # determine option fair value using BSM
        if self.option_type == "call":
            fair_value = BS_CALL(S, K, T, r, sigma)
        else:
            fair_value = BS_PUT(S, K, T, r, sigma)

        if option_price > fair_value + self.threshold:
            self.go_short(state)
        elif option_price < fair_value - self.threshold:
            self.go_long(state)

    def save(self) -> JSON:
        return list(self.price_history)

    def load(self, data: JSON) -> None:
        if data is not None:
            self.price_history = deque(data, maxlen=self.history_length)
        else:
            self.price_history = deque(maxlen=self.history_length)


class ResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000


class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((popular_buy_price + popular_sell_price) / 2)


class InkStrategy(Strategy):
    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        window_size: int = 50,  # mid prices to compute MA
        threshold: int = 2,  # min abs diff between current and MA
        trade_unit: int = 5,  # qty per signal
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

        # compute curr mid px using the best bid/ask
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = round((best_bid + best_ask) / 2)

        # update MA
        self.price_window.append(mid_price)
        if len(self.price_window) < self.price_window.maxlen:
            # enough samples check
            return

        moving_average = sum(self.price_window) / len(self.price_window)
        deviation = mid_price - moving_average

        # get curr position
        position = state.position.get(self.symbol, 0)

        # if px higher than MA, sell
        if deviation > self.threshold and position > -self.limit:
            # calculate qty without breaching limit
            quantity = min(self.trade_unit, position + self.limit)
            if quantity > 0:
                self.sell(mid_price, quantity)

        # if px lower than MA, buy
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


class PicnicBasket1Strategy(BasketStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        basket_weights = {
            "CROISSANTS": 6,
            "JAMS": 3,
            "DJEMBES": 1,
        }
        default_spread_mean = 56.92
        default_spread_std = 80.78
        spread_std_window = 30
        zscore_threshold = 2
        target_position = 58

        super().__init__(
            symbol=symbol,
            limit=limit,
            basket_weights=basket_weights,
            default_spread_mean=default_spread_mean,
            default_spread_std=default_spread_std,
            spread_std_window=spread_std_window,
            zscore_threshold=zscore_threshold,
            target_position=target_position,
        )


class PicnicBasket2Strategy(BasketStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        basket_weights = {
            "CROISSANTS": 4,
            "JAMS": 2,
        }
        default_spread_mean = 36.46
        default_spread_std = 52.15
        spread_std_window = 30
        zscore_threshold = 3
        target_position = 88

        super().__init__(
            symbol=symbol,
            limit=limit,
            basket_weights=basket_weights,
            default_spread_mean=default_spread_mean,
            default_spread_std=default_spread_std,
            spread_std_window=spread_std_window,
            zscore_threshold=zscore_threshold,
            target_position=target_position,
        )


class VolcanicVoucher9500Strategy(BlackScholesStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(
            symbol=symbol,
            limit=limit,
            underlying_symbol="VOLCANIC_ROCK",
            strike_price=9500,
            initial_time_to_expiry=(365 - 3) / 365,
            option_type="call",
            threshold=2.0,
        )


class VolcanicVoucher9750Strategy(BlackScholesStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(
            symbol=symbol,
            limit=limit,
            underlying_symbol="VOLCANIC_ROCK",
            strike_price=9750,
            initial_time_to_expiry=(365 - 3) / 365,
            option_type="call",
            threshold=2.0,
        )


class VolcanicVoucher10000Strategy(BlackScholesStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(
            symbol=symbol,
            limit=limit,
            underlying_symbol="VOLCANIC_ROCK",
            strike_price=10000,
            initial_time_to_expiry=(365 - 3) / 365,
            option_type="call",
            threshold=2.0,
        )


class VolcanicVoucher10250Strategy(BlackScholesStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(
            symbol=symbol,
            limit=limit,
            underlying_symbol="VOLCANIC_ROCK",
            strike_price=10250,
            initial_time_to_expiry=(365 - 3) / 365,
            option_type="call",
            threshold=2.0,
        )


class VolcanicVoucher10500Strategy(BlackScholesStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(
            symbol=symbol,
            limit=limit,
            underlying_symbol="VOLCANIC_ROCK",
            strike_price=10500,
            initial_time_to_expiry=(365 - 3) / 365,
            option_type="call",
            threshold=2.0,
        )


class Trader:
    def __init__(self) -> None:
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
        }

        self.strategies: dict[Symbol, Strategy] = {
            symbol: clazz(symbol, limits[symbol])
            for symbol, clazz in {
                "RAINFOREST_RESIN": ResinStrategy,
                "KELP": KelpStrategy,
                "SQUID_INK": InkStrategy,
                # "CROISSANTS": None,
                # "JAMS": None,
                # "DJEMBES": None,
                "PICNIC_BASKET1": PicnicBasket1Strategy,
                "PICNIC_BASKET2": PicnicBasket2Strategy,
                # "VOLCANIC_ROCK": None,
                "VOLCANIC_ROCK_VOUCHER_9500": VolcanicVoucher9500Strategy,
                "VOLCANIC_ROCK_VOUCHER_9750": VolcanicVoucher9750Strategy,
                "VOLCANIC_ROCK_VOUCHER_10000": VolcanicVoucher10000Strategy,
                "VOLCANIC_ROCK_VOUCHER_10250": VolcanicVoucher10250Strategy,
                "VOLCANIC_ROCK_VOUCHER_10500": VolcanicVoucher10500Strategy,
            }.items()
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if symbol in state.order_depths:
                strategy_orders, strategy_conversions = strategy.run(state)
                orders[symbol] = strategy_orders
                conversions += strategy_conversions

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
