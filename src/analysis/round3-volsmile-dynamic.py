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
from typing import Any, TypeAlias

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
                observation.sunlight,
                observation.humidity,
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
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


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
        """Calculate the volume-weighted mid price"""
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(self, state: TradingState) -> OrderDepth:
        """Calculate synthetic basket order depth from components"""
        order_depths = state.order_depths
        basket_composition = self.basket_weights

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Track the best bids and asks for each component in the basket
        component_best_bids = {}
        component_best_asks = {}

        # Calculate the best bid and ask for each component
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

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = sum(
            component_best_bids[component] * weight
            for component, weight in basket_composition.items()
        )
        implied_ask = sum(
            component_best_asks[component] * weight
            for component, weight in basket_composition.items()
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            # Calculate how many baskets we can create based on each component's volume
            implied_bid_volumes = []
            for component, weight in basket_composition.items():
                if component_best_bids[component] > 0:
                    component_volume = order_depths[component].buy_orders[
                        component_best_bids[component]
                    ]
                    implied_bid_volumes.append(component_volume // weight)

            if implied_bid_volumes:  # Make sure we have volumes to calculate with
                implied_bid_volume = min(implied_bid_volumes)
                synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            # Calculate how many baskets we can create based on each component's volume
            implied_ask_volumes = []
            for component, weight in basket_composition.items():
                if component_best_asks[component] < float("inf"):
                    component_volume = -order_depths[component].sell_orders[
                        component_best_asks[component]
                    ]
                    implied_ask_volumes.append(component_volume // weight)

            if implied_ask_volumes:  # Make sure we have volumes to calculate with
                implied_ask_volume = min(implied_ask_volumes)
                synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def act(self, state: TradingState) -> None:
        # Check if all required products are in order_depths
        basket_type = self.symbol
        if any(
            symbol not in state.order_depths
            for symbol in list(self.basket_weights.keys()) + [basket_type]
        ):
            return

        basket_order_depth = state.order_depths[basket_type]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(state)

        # Check if order depths have sufficient data
        if (
            not basket_order_depth.buy_orders
            or not basket_order_depth.sell_orders
            or not synthetic_order_depth.buy_orders
            or not synthetic_order_depth.sell_orders
        ):
            return

        # Calculate spread
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid

        # Update spread history
        self.spread_history.append(spread)

        # Need enough history for statistical calculation
        if len(self.spread_history) < self.spread_std_window:
            return

        # Calculate z-score
        spread_std = np.std(list(self.spread_history))
        zscore = (spread - self.default_spread_mean) / spread_std

        # Get current position
        position = state.position.get(basket_type, 0)

        # Trading logic based on z-score
        if zscore >= self.zscore_threshold:
            if position > -self.target_position:
                quantity_to_sell = min(
                    position + self.target_position, self.limit + position
                )
                if quantity_to_sell > 0:
                    # Sell at best bid price
                    price = max(basket_order_depth.buy_orders.keys())
                    self.sell(price, quantity_to_sell)

        elif zscore <= -self.zscore_threshold:
            if position < self.target_position:
                quantity_to_buy = min(
                    self.target_position - position, self.limit - position
                )
                if quantity_to_buy > 0:
                    # Buy at best ask price
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


def normal_pdf(x: float) -> float:
    return math.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)


def BS_implied_vol_call(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    initial_guess: float = 0.2,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    sigma = initial_guess
    for i in range(max_iter):
        price = BS_CALL(S, K, T, r, sigma)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
        vega = S * math.sqrt(T) * normal_pdf(d1)
        if vega < 1e-8:
            break  # avoid division by near-zero
        sigma = sigma - diff / vega
    return sigma  # Return last computed sigma if convergence not reached


def BS_implied_vol_put(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    initial_guess: float = 0.2,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    sigma = initial_guess
    for i in range(max_iter):
        price = BS_PUT(S, K, T, r, sigma)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
        vega = S * math.sqrt(T) * normal_pdf(d1)
        if vega < 1e-8:
            break
        sigma = sigma - diff / vega
    return sigma


class VolcanicVoucherSmileStrategy(SignalStrategy):
    def __init__(
        self,
        symbol: Symbol,
        limit: int,
        threshold: float = 0.02,  # IV difference threshold for trading
        underlying_symbol: str = "VOLCANIC_ROCK",
        voucher_symbols: list = None,
        initial_time_to_expiry: float = (365 - 0) / 365,  # in years
    ) -> None:
        super().__init__(symbol, limit)

        # Parameters
        self.threshold = threshold
        self.underlying_symbol = underlying_symbol
        self.initial_time_to_expiry = initial_time_to_expiry

        # Default voucher symbols if none provided
        if voucher_symbols is None:
            self.voucher_symbols = [
                "VOLCANIC_ROCK_VOUCHER_9500",
                "VOLCANIC_ROCK_VOUCHER_9750",
                "VOLCANIC_ROCK_VOUCHER_10000",
                "VOLCANIC_ROCK_VOUCHER_10250",
                "VOLCANIC_ROCK_VOUCHER_10500",
            ]
        else:
            self.voucher_symbols = voucher_symbols

        # Strike prices extracted from symbol names
        self.strike_prices = {
            symbol: int(symbol.split("_")[-1]) for symbol in self.voucher_symbols
        }

        # For tracking metrics
        self.last_ivs = {}
        self.last_fit_params = None
        self.last_mse = None

    def _compute_log_moneyness(self, S: float, K: float, T: float) -> float:
        """Compute the log-moneyness metric m_t = ln(K/S)/sqrt(T)"""
        return math.log(K / S) / math.sqrt(T)

    def _fit_quadratic(self, x_values: list, y_values: list) -> tuple:
        """Fit a quadratic curve to the data points."""
        if len(x_values) < 3:
            # Need at least 3 points for a quadratic fit
            return None, float("inf")

        # Use numpy's polyfit to fit a quadratic curve (degree 2)
        params = np.polyfit(x_values, y_values, 2)

        # Calculate mean squared error of the fit
        fitted_values = [params[0] * x**2 + params[1] * x + params[2] for x in x_values]
        mse = sum(
            (fitted - actual) ** 2 for fitted, actual in zip(fitted_values, y_values)
        ) / len(x_values)

        return params, mse

    def _evaluate_quadratic(self, params: tuple, x: float) -> float:
        """Evaluate the quadratic function at point x."""
        a, b, c = params
        return a * x**2 + b * x + c

    def act(self, state: TradingState) -> None:
        # Check if all required markets have order depths
        if self.underlying_symbol not in state.order_depths:
            return

        # Get underlying price
        underlying_price = self.get_mid_price(state, self.underlying_symbol)

        # Calculate time to expiry (decreasing with each day)
        mill = 1_000_000  # Ticks per day
        frac = state.timestamp / mill  # Fraction of day elapsed (0 to 1)
        T = self.initial_time_to_expiry - (frac / 365)

        # Collect implied volatilities and moneyness for each voucher
        log_moneyness_values = []
        iv_values = []

        for voucher_symbol in self.voucher_symbols:
            if voucher_symbol not in state.order_depths:
                continue

            if (
                not state.order_depths[voucher_symbol].buy_orders
                or not state.order_depths[voucher_symbol].sell_orders
            ):
                continue

            # Get voucher price and strike
            voucher_price = self.get_mid_price(state, voucher_symbol)
            strike_price = self.strike_prices[voucher_symbol]

            # Calculate implied volatility
            try:
                iv = BS_implied_vol_call(
                    S=underlying_price,
                    K=strike_price,
                    T=T,
                    r=0,
                    market_price=voucher_price,
                )

                # Calculate log-moneyness
                moneyness = self._compute_log_moneyness(
                    underlying_price, strike_price, T
                )

                # Store the values
                log_moneyness_values.append(moneyness)
                iv_values.append(iv)

                # Save this IV
                self.last_ivs[voucher_symbol] = iv

            except Exception:
                # Skip this voucher if IV calculation fails
                continue

        # Make sure we have enough data points for a meaningful fit
        if len(log_moneyness_values) < 3:
            return

        # Fit the volatility smile curve (quadratic)
        fit_params, mse = self._fit_quadratic(log_moneyness_values, iv_values)

        if fit_params is None:
            return

        # Save the fit parameters and MSE
        self.last_fit_params = fit_params
        self.last_mse = mse

        # Now analyze our specific voucher
        if self.symbol not in state.order_depths:
            return

        if (
            not state.order_depths[self.symbol].buy_orders
            or not state.order_depths[self.symbol].sell_orders
        ):
            return

        # Get our voucher price and strike
        our_price = self.get_mid_price(state, self.symbol)
        our_strike = self.strike_prices[self.symbol]

        # Get our actual IV
        our_iv = BS_implied_vol_call(
            S=underlying_price, K=our_strike, T=T, r=0, market_price=our_price
        )

        # Get our log-moneyness
        our_moneyness = self._compute_log_moneyness(underlying_price, our_strike, T)

        # Get the fitted IV for our moneyness
        fitted_iv = self._evaluate_quadratic(fit_params, our_moneyness)

        # Trading logic based on IV mispricing
        iv_diff = our_iv - fitted_iv

        if iv_diff > self.threshold:
            # Actual IV is higher than fitted IV - option is overpriced
            self.go_short(state)
        elif iv_diff < -self.threshold:
            # Actual IV is lower than fitted IV - option is underpriced
            self.go_long(state)

    def save(self) -> JSON:
        return {
            "last_ivs": self.last_ivs,
            "last_fit_params": (
                self.last_fit_params.tolist()
                if self.last_fit_params is not None
                and hasattr(self.last_fit_params, "tolist")
                else self.last_fit_params
            ),
            "last_mse": float(self.last_mse) if self.last_mse is not None else None,
        }

    def load(self, data: JSON) -> None:
        if data is not None:
            self.last_ivs = data.get("last_ivs", {})
            self.last_fit_params = data.get("last_fit_params")
            self.last_mse = data.get("last_mse")


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
                # "CROISSANTS": PicnicBasket1Strategy,
                # "JAMS": PicnicBasket1Strategy,
                # "DJEMBES": PicnicBasket1Strategy,
                "PICNIC_BASKET1": PicnicBasket1Strategy,
                "PICNIC_BASKET2": PicnicBasket2Strategy,
                # "VOLCANIC_ROCK": NoStrategy,
                "VOLCANIC_ROCK_VOUCHER_9500": VolcanicVoucherSmileStrategy,
                "VOLCANIC_ROCK_VOUCHER_9750": VolcanicVoucherSmileStrategy,
                "VOLCANIC_ROCK_VOUCHER_10000": VolcanicVoucherSmileStrategy,
                "VOLCANIC_ROCK_VOUCHER_10250": VolcanicVoucherSmileStrategy,
                "VOLCANIC_ROCK_VOUCHER_10500": VolcanicVoucherSmileStrategy,
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
