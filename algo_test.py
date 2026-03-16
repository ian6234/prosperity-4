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
                weighted_mid += price_level * asks[price_level]
                total_volume += asks[price_level]
        if len(bids) != 0:
            for price_level in bids:
                weighted_mid += price_level * bids[price_level]
                total_volume += bids[price_level]
        weighted_mid = weighted_mid / total_volume
        return weighted_mid

    # Signal/Pricing Logic
    def emerald_price(self):
        return 10000

    def tomato_price(self, state):
        return self.volume_weighted_mid("TOMATOES", state)


    # Execution Logic
    def basic_execution(self, product, pos_limit, position, fair_value, state: TradingState):

        order_depth = state.order_depths[product]
        orders = []

        last_ask = -1
        last_bid = -1
        threshold = 0
        new_position = position

        if len(order_depth.sell_orders) != 0:
            buy_quantity = 0

            for ask, quantity in list(order_depth.sell_orders.items()):
                if float(ask) + threshold - fair_value < 0:
                    buy_quantity += quantity
                    if -buy_quantity < (pos_limit - position):
                        last_ask = ask
            buy_quantity = min(-buy_quantity, pos_limit - position)

            if last_ask != -1:
                print(f"BUY {product}", str(buy_quantity) + "x", last_ask)
                orders.append(Order(product, last_ask, buy_quantity))
                new_position = position + buy_quantity

        if len(order_depth.buy_orders) != 0:
            sell_quantity = 0

            for bid, quantity in list(order_depth.buy_orders.items()):
                if float(bid) - threshold - fair_value > 0:
                    sell_quantity += quantity
                    if sell_quantity < (pos_limit + position):
                        last_bid = bid

            sell_quantity = min(sell_quantity, pos_limit + position)

            if last_bid != -1:
                print(f"SELL {product}", str(-sell_quantity) + "x", last_bid)
                orders.append(Order(product, last_bid, -sell_quantity))
                new_position = position - sell_quantity

        return orders, new_position


        return orders
    def bid(self):
        return 15

    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        pos_limits = {"EMERALDS": 80, "TOMATOES": 80}

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            current_position = state.position[product]

            fair_value = 10
            if product == "EMERALDS":
                fair_value = self.emerald_price()  # Participant should calculate this value
            elif product == "TOMATOES":
                fair_value = self.tomato_price(state)

            print("Acceptable price : " + str(fair_value))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(
                len(order_depth.sell_orders)))

            # Order Execution logic
            # Basic logic: clear the market, then place resting orders.
            orders, new_position = self.basic_execution(product, pos_limits[product], current_position, fair_value, state)

            result[product] = orders

        traderData = ""  # No state needed - we check position directly
        conversions = 0
        return result, conversions, traderData