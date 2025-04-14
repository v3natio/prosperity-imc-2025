from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any, Deque, Optional
import string
import jsonpickle
import numpy as np
import math
from collections import deque
from abc import abstractmethod


class Product:
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"


# Updated PARAMS to include separate parameters for each basket's spread
PARAMS = {
    Product.PICNIC_BASKET1: {
        "default_spread_mean": 56.92,
        "default_spread_std": 80.78,
        "spread_std_window": 30,
        "zscore_threshold": 2,
        "target_position": 58,
    },
    Product.PICNIC_BASKET2: {
        "default_spread_mean": 36.46,
        "default_spread_std": 52.15,
        "spread_std_window": 30,
        "zscore_threshold": 3,
        "target_position": 88,
    },
}

# Basket compositions
BASKET_WEIGHTS = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1,
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2,
    },
}


class Trader:
    def __init__(self, params=None, trade_synthetic=False):
        if params is None:
            params = PARAMS
        self.params = params
        self.trade_synthetic = trade_synthetic  # flag

        self.LIMIT = {
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.KELP: 50,
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50,
        }

        # Initialize strategies
        self.kelp_strategy = KelpStrategy(Product.KELP, self.LIMIT[Product.KELP])
        self.resin_strategy = ResinStrategy(
            Product.RAINFOREST_RESIN, self.LIMIT[Product.RAINFOREST_RESIN]
        )
        self.ink_strategy = InkStrategy(
            Product.SQUID_INK, self.LIMIT[Product.SQUID_INK]
        )

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth], basket_type: str
    ) -> OrderDepth:
        # Get the basket composition
        basket_composition = BASKET_WEIGHTS[basket_type]

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

    def convert_synthetic_basket_orders(
        self,
        synthetic_orders: List[Order],
        order_depths: Dict[str, OrderDepth],
        basket_type: str,
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {component: [] for component in BASKET_WEIGHTS[basket_type]}

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths, basket_type
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

        # Get the basket composition
        basket_composition = BASKET_WEIGHTS[basket_type]

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                component_prices = {
                    component: min(order_depths[component].sell_orders.keys())
                    for component in basket_composition
                    if component in order_depths and order_depths[component].sell_orders
                }
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                component_prices = {
                    component: max(order_depths[component].buy_orders.keys())
                    for component in basket_composition
                    if component in order_depths and order_depths[component].buy_orders
                }
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            for component, weight in basket_composition.items():
                if component in component_prices:
                    component_order = Order(
                        component,
                        component_prices[component],
                        quantity * weight,
                    )
                    component_orders[component].append(component_order)

        return component_orders

    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
        basket_type: str,
    ):
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[basket_type]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(
            order_depths, basket_type
        )

        if target_position > basket_position:
            if (
                not basket_order_depth.sell_orders
                or not synthetic_order_depth.buy_orders
            ):
                return None

            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(basket_type, basket_ask_price, execute_volume)]
            if self.trade_synthetic:  # flag
                synthetic_orders = [
                    Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
                ]
                aggregate_orders = self.convert_synthetic_basket_orders(
                    synthetic_orders, order_depths, basket_type
                )
                aggregate_orders[basket_type] = basket_orders
            else:
                aggregate_orders = {basket_type: basket_orders}

            return aggregate_orders

        else:
            if (
                not basket_order_depth.buy_orders
                or not synthetic_order_depth.sell_orders
            ):
                return None

            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(basket_type, basket_bid_price, -execute_volume)]
            if self.trade_synthetic:  # flag
                synthetic_orders = [
                    Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
                ]
                aggregate_orders = self.convert_synthetic_basket_orders(
                    synthetic_orders, order_depths, basket_type
                )
                aggregate_orders[basket_type] = basket_orders
            else:
                aggregate_orders = {basket_type: basket_orders}

            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        basket_type: str,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if basket_type not in order_depths.keys():
            return None

        basket_order_depth = order_depths[basket_type]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(
            order_depths, basket_type
        )

        # Check if order depths have sufficient data
        if (
            not basket_order_depth.buy_orders
            or not basket_order_depth.sell_orders
            or not synthetic_order_depth.buy_orders
            or not synthetic_order_depth.sell_orders
        ):
            return None

        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid

        # Initialize the spread history if needed
        if f"{basket_type}_spread_history" not in spread_data:
            spread_data[f"{basket_type}_spread_history"] = []

        spread_data[f"{basket_type}_spread_history"].append(spread)

        spread_std_window = self.params[basket_type]["spread_std_window"]

        if len(spread_data[f"{basket_type}_spread_history"]) < spread_std_window:
            return None
        elif len(spread_data[f"{basket_type}_spread_history"]) > spread_std_window:
            spread_data[f"{basket_type}_spread_history"].pop(0)

        spread_std = np.std(spread_data[f"{basket_type}_spread_history"])

        # Use basket-specific parameters
        zscore = (spread - self.params[basket_type]["default_spread_mean"]) / spread_std

        if zscore >= self.params[basket_type]["zscore_threshold"]:
            if basket_position != -self.params[basket_type]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[basket_type]["target_position"],
                    basket_position,
                    order_depths,
                    basket_type,
                )

        if zscore <= -self.params[basket_type]["zscore_threshold"]:
            if basket_position != self.params[basket_type]["target_position"]:
                return self.execute_spread_orders(
                    self.params[basket_type]["target_position"],
                    basket_position,
                    order_depths,
                    basket_type,
                )

        spread_data[f"{basket_type}_prev_zscore"] = zscore
        return None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        # Run KELP strategy
        if Product.KELP in state.order_depths:
            kelp_orders = self.kelp_strategy.run(state)
            if kelp_orders:
                result[Product.KELP] = kelp_orders

        # Run RESIN strategy
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_orders = self.resin_strategy.run(state)
            if resin_orders:
                result[Product.RAINFOREST_RESIN] = resin_orders

        # Run INK strategy
        if Product.SQUID_INK in state.order_depths:
            ink_orders = self.ink_strategy.run(state)
            if ink_orders:
                result[Product.SQUID_INK] = ink_orders

        # Initialize spread data if needed
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                # Common spread data structure for all baskets
                # Individual basket data will be stored with basket-specific keys
                "clear_flag": False,
                "curr_avg": 0,
            }

        # Handle PICNIC_BASKET1 spread strategy
        if Product.PICNIC_BASKET1 in state.order_depths:
            basket_position = (
                state.position[Product.PICNIC_BASKET1]
                if Product.PICNIC_BASKET1 in state.position
                else 0
            )
            spread_orders = self.spread_orders(
                state.order_depths,
                Product.PICNIC_BASKET1,
                basket_position,
                traderObject[Product.SPREAD],
            )
            if spread_orders != None:
                if self.trade_synthetic:
                    for component in BASKET_WEIGHTS[Product.PICNIC_BASKET1]:
                        if component in spread_orders:
                            if component in result:
                                result[component].extend(
                                    spread_orders.get(component, [])
                                )
                            else:
                                result[component] = spread_orders.get(component, [])
                result[Product.PICNIC_BASKET1] = spread_orders.get(
                    Product.PICNIC_BASKET1, []
                )

        # Handle PICNIC_BASKET2 spread strategy
        if Product.PICNIC_BASKET2 in state.order_depths:
            basket_position = (
                state.position[Product.PICNIC_BASKET2]
                if Product.PICNIC_BASKET2 in state.position
                else 0
            )
            spread_orders = self.spread_orders(
                state.order_depths,
                Product.PICNIC_BASKET2,
                basket_position,
                traderObject[Product.SPREAD],
            )
            if spread_orders != None:
                if self.trade_synthetic:
                    for component in BASKET_WEIGHTS[Product.PICNIC_BASKET2]:
                        if component in spread_orders:
                            if component in result:
                                result[component].extend(
                                    spread_orders.get(component, [])
                                )
                            else:
                                result[component] = spread_orders.get(component, [])
                result[Product.PICNIC_BASKET2] = spread_orders.get(
                    Product.PICNIC_BASKET2, []
                )

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData


class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> Any:
        return None

    def load(self, data: Any) -> None:
        pass


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

    def save(self) -> Any:
        return list(self.window)

    def load(self, data: Any) -> None:
        if data is not None:
            self.window = deque(data)
        else:
            self.window = deque()


class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((popular_buy_price + popular_sell_price) / 2)


class ResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000


class InkStrategy(Strategy):
    def __init__(
        self,
        symbol: str,
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

    def save(self) -> Any:
        return list(self.price_window)

    def load(self, data: Any) -> None:
        if data is not None:
            self.price_window = deque(data, maxlen=self.price_window.maxlen)
        else:
            self.price_window = deque(maxlen=self.price_window.maxlen)
