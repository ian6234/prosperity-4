import math

import jsonpickle
import numpy as np

from datamodel import OrderDepth, TradingState, Order
from typing import List
import string


# Helper functions
# finds the volume weighted mid-price
def volume_weighted_mid(product, state: TradingState) -> float:
    order_depth = state.order_depths[product]
    bids = order_depth.buy_orders
    asks = order_depth.sell_orders
    weighted_mid = 0
    total_volume = 0
    if len(asks) != 0:
        for price_level in asks:
            weighted_mid += price_level * abs(asks[price_level])
            total_volume += abs(asks[price_level])
    if len(bids) != 0:
        for price_level in bids:
            weighted_mid += price_level * bids[price_level]
            total_volume += bids[price_level]
    if total_volume == 0:
        return -1
    weighted_mid = weighted_mid / total_volume
    return weighted_mid


# tries to find the 'true' mid by using the mid of the deepest bid/ask
def true_mid(product, state: TradingState) -> float:
    order_depth = state.order_depths[product]
    bids = order_depth.buy_orders
    asks = order_depth.sell_orders

    min_bid = min([x for x in bids])
    max_ask = max([x for x in asks])

    return (min_bid + max_ask) / 2

# the building block for every strategy, stores what products are traded on it and the book/position state being traded on.
class BaseStrategy:

    def __init__(self, name: str, products: List[str]):
        self.name = name
        self.products = products

    def generate(self, state: TradingState,
                 implied_positions: dict,
                 order_budgets: dict,
                 arb_positions: dict,
                 resting_book: dict):
        return 0


class SingleProductStrategy(BaseStrategy):

    def __init__(self, symbol: str, pos_limit: int):
        super().__init__(name=symbol, products=[symbol])
        self.symbol = symbol
        self.pos_limit = pos_limit
        self.max_threshold = 1

    # override these three below with product-specific logic
    def fair_value(self, state: TradingState) -> float:
        return -1

    def edge(self) -> float:
        # base edge for MM
        return 6

    def thresholds(self, position: int):
        pos_fraction = position / self.pos_limit

        # threshold logic: reduces to 0 / increases to twice the max threshold at either position limit.
        buy_threshold = round(self.max_threshold * (1 + pos_fraction))
        sell_threshold = round(self.max_threshold * (1 - pos_fraction))

        return buy_threshold, sell_threshold

    # taker logic
    def taker_execution(self, fair_value, buy_threshold, sell_threshold, order_budgets, implied_positions, resting_book):

        product = self.symbol
        position = implied_positions[product]

        orders = []

        remaining_bids = resting_book[product]['bids']
        remaining_asks = resting_book[product]['asks']

        last_ask = -1
        last_bid = -1

        max_buy_qty = order_budgets[product]['buy']
        max_sell_qty = order_budgets[product]['sell']

        # Taker Logic

        best_bid = max([x for x in remaining_bids])
        best_ask = min([x for x in remaining_asks])

        would_buy = False
        would_sell = False
        if best_ask + buy_threshold <= fair_value:
            would_buy = True
        if best_bid - sell_threshold >= fair_value:
            would_sell = True

        if would_buy and would_sell:
            if position > 0:
                would_buy = False  # already long, prefer to sell
            elif position < 0:
                would_sell = False  # already short, prefer to buy
            else:
                would_buy = False  # neutral, default to selling
                # (corrects for structural long bias)

        if would_buy:
            buy_quantity = 0
            # sorted creates a new list - so safe to delete from remaining asks
            for ask, quantity in sorted(remaining_asks.items()):
                # if ask below fair value, buy
                if float(ask) + buy_threshold - fair_value <= 0:
                    # if current quantity exceeds position limit, stop.
                    capacity = max_buy_qty - abs(buy_quantity)
                    if capacity <= 0:
                        break

                    take_quantity = min(capacity, abs(quantity))
                    buy_quantity += take_quantity

                    last_ask = ask

                    remaining_asks[ask] += take_quantity
                    if remaining_asks[ask] == 0:
                        del remaining_asks[ask]

                # otherwise stop
                else:
                    break

            if last_ask != -1:
                #   print(f"BUY {product}", str(buy_quantity) + "x", last_ask)
                orders.append(Order(product, last_ask, buy_quantity))
                order_budgets[product]['buy'] -= buy_quantity
                implied_positions[product] += buy_quantity

        elif would_sell:
            sell_quantity = 0
            for bid, quantity in sorted(remaining_bids.items(), reverse=True):
                # if bid above fair value, sell
                if float(bid) - sell_threshold - fair_value >= 0:
                    # if current quantity exceeds position limit, stop.
                    capacity = max_sell_qty - abs(sell_quantity)
                    if capacity <= 0:
                        break

                    take_quantity = min(capacity, abs(quantity))
                    sell_quantity += take_quantity

                    last_bid = bid

                    remaining_bids[bid] -= take_quantity
                    if remaining_bids[bid] == 0:
                        del remaining_bids[bid]
                # otherwise stop
                else:
                    break

            if last_bid != -1:
                #    print(f"SELL {product}", str(-sell_quantity) + "x", last_bid)
                orders.append(Order(product, last_bid, -sell_quantity))
                order_budgets[product]['sell'] -= sell_quantity
                implied_positions[product] += sell_quantity
        return orders

    # Maker Logic
    def market_make(self, fair_value, order_budgets, resting_book):
        orders = []
        # Maker Logic
        product = self.symbol
        edge = self.edge()

        remaining_bids = resting_book[product]['bids']
        remaining_asks = resting_book[product]['asks']

        bid_qty = order_budgets[product]['buy']
        ask_qty = order_budgets[product]['sell']

        if ask_qty > 0:
            if len(remaining_asks) != 0:

                last_ask = min(remaining_asks.keys())
                if last_ask - 1 > fair_value:
                    orders.append(Order(product, last_ask - 1, -ask_qty))
                else:
                    orders.append(Order(product, round(fair_value + edge), -ask_qty))

            else:
                orders.append(Order(product, round(fair_value + edge), -ask_qty))
        if bid_qty > 0:
            if len(remaining_bids) != 0:

                last_bid = max(remaining_bids.keys())

                if last_bid + 1 < fair_value:
                    orders.append(Order(product, last_bid + 1, bid_qty))
                else:
                    orders.append(Order(product, round(fair_value - edge), bid_qty))
            else:
                orders.append(Order(product, round(fair_value - edge), bid_qty))

        return orders

    def generate(self, state: TradingState,
                 implied_positions: dict,
                 order_budgets: dict,
                 arb_positions: dict,
                 resting_book: dict):

        implied_position = implied_positions.get(self.symbol, 0)

        fair_value = self.fair_value(state)

        buy_threshold, sell_threshold = self.thresholds(implied_position)

        # first taking execution on any orders
        instant_orders = self.taker_execution(fair_value, buy_threshold, sell_threshold, order_budgets, implied_positions, resting_book)

        # then making on the remaining book
        resting_orders = self.market_make(fair_value, order_budgets, resting_book)

        orders = instant_orders
        orders.extend(resting_orders)
        return orders


class MultiProductStrategy(BaseStrategy):
    def __init__(self, name: str, products: List[str]):
        super().__init__(name, products)

# Multi Product Strategies


# Single Product Strategies

class Emeralds(SingleProductStrategy):
    def __init__(self, symbol: str, pos_limit: int):
        super().__init__(symbol, pos_limit)

    def fair_value(self, state: TradingState) -> float:
        return 10000


class Tomatoes(SingleProductStrategy):
    def __init__(self, symbol: str, pos_limit: int):
        super().__init__(symbol, pos_limit)
        self.max_threshold = 2

    def fair_value(self, state: TradingState) -> float:
        return true_mid(self.symbol, state)



class Trader:

    def bid(self):
        return 15

    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""
        # get new trader data
        trader_data = jsonpickle.decode(state.traderData) if state.traderData else {
            "EMERALDS": {},
            "TOMATOES": {},
            "debug": {},
            "arb_positions": {}
        }

        # set up position tracking
        pos_limits = {"EMERALDS": 80, "TOMATOES": 80}
        raw_positions = {}
        order_budgets = {}
        resting_book = {}
        for product in state.order_depths:
            # get positions and calculate order budgets
            raw_positions[product] = state.position.get(product, 0)
            order_budgets[product] = {
                'buy': pos_limits[product] - raw_positions[product],
                'sell': pos_limits[product] + raw_positions[product]
            }

            # create a mutable resting book
            resting_book[product] = {
                "bids": dict(state.order_depths[product].buy_orders),
                "asks": dict(state.order_depths[product].sell_orders),
            }

        implied_positions = dict(raw_positions)

        # create strategy objects
        strategies = [
            Emeralds("EMERALDS", 80),
            Tomatoes("TOMATOES", 80),
        ]

        # Orders to be placed on exchange matching engine
        result = {}

        for strategy in strategies:
            result[strategy.symbol] = strategy.generate(state, implied_positions, order_budgets,
                                                        trader_data['arb_positions'], resting_book)

        trader_data_encode = jsonpickle.encode(trader_data)

        conversions = 0
        return result, conversions, trader_data_encode