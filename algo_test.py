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


# Product Specific logic
class BaseProduct():

    def __init__(self, symbol: str, pos_limit: int):
        self.symbol = symbol
        self.pos_limit = pos_limit
        self.max_threshold = 1

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

    # Taker Logic
    def taker_execution(self, position, fair_value, buy_threshold, sell_threshold, state: TradingState):

        product = self.symbol
        pos_limit = self.pos_limit

        order_depth = state.order_depths[product]
        orders = []

        remaining_bids = dict(order_depth.buy_orders)
        remaining_asks = dict(order_depth.sell_orders)

        last_ask = -1
        last_bid = -1


        position_delta = 0

        # Taker Logic

        best_bid = max([x for x in order_depth.buy_orders])
        best_ask = min([x for x in order_depth.sell_orders])

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
            max_quantity = pos_limit - position
            for ask, quantity in sorted(order_depth.sell_orders.items()):
                # if ask below fair value, buy
                if float(ask) + buy_threshold - fair_value <= 0:
                    # if current quantity exceeds position limit, stop.
                    capacity = max_quantity - abs(buy_quantity)
                    if capacity <= 0:
                        break

                    take_quantity = min(capacity, abs(quantity))
                    buy_quantity += take_quantity

                    last_ask = ask

                    remaining_asks[ask] -= take_quantity
                    if remaining_asks[ask] == 0:
                        del remaining_asks[ask]

                # otherwise stop
                else:
                    break

            if last_ask != -1:
                #   print(f"BUY {product}", str(buy_quantity) + "x", last_ask)
                orders.append(Order(product, last_ask, buy_quantity))
                position_delta = buy_quantity

        elif would_sell:
            sell_quantity = 0
            max_quantity = pos_limit + position
            for bid, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
                # if bid above fair value, sell
                if float(bid) - sell_threshold - fair_value >= 0:
                    # if current quantity exceeds position limit, stop.
                    capacity = max_quantity - abs(sell_quantity)
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
                position_delta = -sell_quantity

        return orders, position_delta, remaining_bids, remaining_asks

    # Maker Logic
    def market_make(self, position, position_delta, fair_value, remaining_bids, remaining_asks):
        orders = []
        # Maker Logic
        product = self.symbol
        pos_limit = self.pos_limit
        edge = self.edge()

        # extra logic needed to avoid breaching the exchange's weird position limits
        # if taker bought, ask can only be as big as the original position allowed, and vice versa
        if position_delta >= 0:
            ask_qty = pos_limit + position
            bid_qty = pos_limit - (position + position_delta)
        else:
            ask_qty = pos_limit + (position + position_delta)
            bid_qty = pos_limit - position


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

# Specific product behaviour (fair value, execution)

class Emeralds(BaseProduct):
    def __init__(self, symbol: str, pos_limit: int):
        super().__init__(symbol, pos_limit)

    def fair_value(self, state: TradingState) -> float:
        return 10000


class Tomatoes(BaseProduct):
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

        pos_limits = {"EMERALDS": 80, "TOMATOES": 80}

        product_objects = {
            "EMERALDS": Emeralds("EMERALDS", 80),
            "TOMATOES": Tomatoes("TOMATOES", 80)
        }

        if state.timestamp == 0:

            trader_data = {
                "EMERALDS": {},
                "TOMATOES": {
                },
            }
        else:
            trader_data = jsonpickle.decode(state.traderData)


        # Orders to be placed on exchange matching engine
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            current_position = state.position.get(product, 0)

            # load the correct pricing/execution for this product
            product_manager = product_objects[product]

            fair_value = product_manager.fair_value(state)

            buy_threshold, sell_threshold = product_manager.thresholds(current_position)

            # first taking execution on any orders
            instant_orders, position_delta, remaining_bids, remaining_asks = product_manager.taker_execution(
                current_position, fair_value, buy_threshold, sell_threshold, state)

            # then making on the remaining book
            resting_orders = product_manager.market_make(current_position, position_delta, fair_value, remaining_bids, remaining_asks)

            result[product] = instant_orders
            result[product].extend(resting_orders)

        trader_data_encode = jsonpickle.encode(trader_data)

        conversions = 0
        return result, conversions, trader_data_encode