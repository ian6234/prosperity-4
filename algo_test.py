import math

import jsonpickle
import numpy as np

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string


class Trader:


    # Helper functions
    def volume_weighted_mid(self, product, state: TradingState) -> float:
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

    # Signal/Pricing Logic
    def emerald_price(self):
        return 10000

    def tomato_price(self, state):

        return self.volume_weighted_mid("TOMATOES", state)


    # Execution Logic
    def basic_execution(self, product, pos_limit, position, fair_value, max_threshold, state: TradingState):

        order_depth = state.order_depths[product]
        orders = []

        remaining_bids = dict(order_depth.buy_orders)
        remaining_asks = dict(order_depth.sell_orders)

        last_ask = -1
        last_bid = -1

        new_position = position

        # create thresholds (skewing) based on current position vs limit.
        # if at positive pos limit, selling threshold drops to 0.
        if position >= 0:
            buy_threshold = round(max_threshold * (1 + abs(position) / pos_limit))
            sell_threshold = round(max_threshold * (1 - abs(position) / pos_limit))
        else:
            buy_threshold = round(max_threshold * (1 - abs(position) / pos_limit))
            sell_threshold = round(max_threshold * (1 + abs(position) / pos_limit))


        # Taker Logic

        if len(order_depth.sell_orders) != 0:
            buy_quantity = 0
            max_quantity = pos_limit - position
            for ask, quantity in list(order_depth.sell_orders.items()):
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
                new_position = position + buy_quantity

        if len(order_depth.buy_orders) != 0:
            sell_quantity = 0
            max_quantity = pos_limit + position
            for bid, quantity in list(order_depth.buy_orders.items()):
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
                new_position = position - sell_quantity

        return orders, new_position, remaining_bids, remaining_asks

    def market_make(self, product, pos_limit, new_position, fair_value, remaining_bids, remaining_asks, edge):
        orders = []
        # Maker Logic

        if len(remaining_asks) != 0:

            last_ask, amount = list(remaining_asks.items())[0]
            if last_ask - 1 > fair_value:
                ask_quantity = pos_limit + new_position
                #    print(f"ASK {product}", str(-ask_quantity) + "x", last_ask - 1)
                orders.append(Order(product, last_ask - 1, -ask_quantity))

        else:
            ask_quantity = pos_limit + new_position
            orders.append(Order(product, fair_value + edge, -ask_quantity))

        if len(remaining_bids) != 0:

            last_bid, amount = list(remaining_bids.items())[0]
            if last_bid + 1 < fair_value:
                bid_quantity = pos_limit - new_position
                #   print(f"BID {product}", str(bid_quantity) + "x", last_bid + 1)
                orders.append(Order(product, last_bid + 1, bid_quantity))
        else:
            bid_quantity = pos_limit - new_position
            orders.append(Order(product, fair_value - edge, bid_quantity))
        return orders

    def bid(self):
        return 15

    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        pos_limits = {"EMERALDS": 80, "TOMATOES": 80}

       # print("traderData: " + state.traderData)
      #  print("Observations: " + str(state.observations))


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

            edge = 1
            threshold = 0

            fair_value = 10
            if product == "EMERALDS":
                fair_value = self.emerald_price()  # Participant should calculate this value
                edge = 7
                threshold = 1
            elif product == "TOMATOES":
                fair_value = self.tomato_price(state)
                fair_value = round(fair_value)
                edge = 6
                threshold = 1

           # print("Acceptable price : " + str(fair_value))
           # print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(
           #     len(order_depth.sell_orders)))

            # Order Execution logic
            # Basic logic: clear the market, then place resting orders.
            instant_orders, new_position, remaining_bids, remaining_asks = self.basic_execution(product, pos_limits[product], current_position, fair_value, threshold, state)
            resting_orders = self.market_make(product, pos_limits[product], new_position, fair_value, remaining_bids, remaining_asks, edge)


            result[product] = instant_orders
            result[product].extend(resting_orders)

        trader_data_encode = jsonpickle.encode(trader_data)

        conversions = 0
        return result, conversions, trader_data_encode