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


# gets the spread on a product
def get_spread(product, state: TradingState) -> float:
    order_depth = state.order_depths[product]
    if order_depth == -1:
        return -1
    bids = order_depth.buy_orders
    asks = order_depth.sell_orders

    if bids and asks:
        max_bid = max([x for x in bids])
        min_ask = min([x for x in asks])
        return min_ask - max_bid
    else:
        return -1

# tries to find the 'true' mid by using the mid of the deepest bid/ask
def true_mid(product, state: TradingState) -> float:
    order_depth = state.order_depths.get(product, -1)
    if order_depth == -1:
        return -1
    bids = order_depth.buy_orders
    asks = order_depth.sell_orders

    if bids and asks:
        min_bid = min([x for x in bids])
        max_ask = max([x for x in asks])
        return (min_bid + max_ask) / 2
    else:
        return -1

# returns top of book mid price
def top_mid(product, state: TradingState) -> float:
    order_depth = state.order_depths.get(product, -1)
    if order_depth == -1:
        return -1
    bids = order_depth.buy_orders
    asks = order_depth.sell_orders

    if bids and asks:
        best_bid = max([x for x in bids])
        best_ask = min([x for x in asks])
        return (best_bid + best_ask) / 2
    else:
        return -1

def simple_moving_average(prices):
    return sum(prices)/len(prices)

def exp_moving_average(last_price, last_ema, timespan) -> float:
    ema = last_price * 2/(1+timespan) + last_ema * (1 - (2/(1+timespan)))
    return ema

# the building block for every strategy, stores what products are traded on it and the book/position state being traded on.
class BaseStrategy:

    def __init__(self, name: str, products: List[str]):
        self.name = name
        self.products = products

    # taker logic - every strategy needs logic to place limit orders.
    def taker_execution(self, product, fair_value, buy_threshold, sell_threshold, order_budgets, implied_positions,
                        resting_book, max_quantity=-1):

        position = implied_positions[product]

        orders = []

        remaining_bids = resting_book[product]['bids']
        remaining_asks = resting_book[product]['asks']

        last_ask = -1
        last_bid = -1

        if max_quantity != -1:
            max_buy_qty = min(order_budgets[product]['buy'], max_quantity)
            max_sell_qty = min(order_budgets[product]['sell'], max_quantity)
        else:
            max_buy_qty = order_budgets[product]['buy']
            max_sell_qty = order_budgets[product]['sell']

        # Taker Logic
        would_buy = False
        would_sell = False

        if remaining_bids:
            best_bid = max([x for x in remaining_bids])
            if best_bid - sell_threshold >= fair_value:
                would_sell = True

        if remaining_asks:
            best_ask = min([x for x in remaining_asks])
            if best_ask + buy_threshold <= fair_value:
                would_buy = True

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

    # every strategy needs to generate orders. overridden
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
        self.max_maker_size = 200

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

    # Maker Logic
    def market_make(self, fair_value, order_budgets, resting_book):
        orders = []
        # Maker Logic
        product = self.symbol
        edge = self.edge()

        remaining_bids = resting_book[product]['bids']
        remaining_asks = resting_book[product]['asks']

        bid_qty = min(self.max_maker_size, order_budgets[product]['buy'])
        ask_qty = min(self.max_maker_size, order_budgets[product]['sell'])

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
        instant_orders = self.taker_execution(self.symbol, fair_value, buy_threshold, sell_threshold, order_budgets, implied_positions, resting_book)

        # then making on the remaining book
        resting_orders = self.market_make(fair_value, order_budgets, resting_book)

        orders = instant_orders
        orders.extend(resting_orders)
        return orders


class MultiProductStrategy(BaseStrategy):
    def __init__(self, name: str, products: List[str]):
        super().__init__(name, products)

# Multi Product Strategies



class BuyNHold(MultiProductStrategy):
    """
    Trend-following buy-and-hold strategy for products with known directional bias.

    On every tick, attempts taker execution toward the target position for each
    product until the position limit is reached. After that, order_budgets are
    exhausted and nothing fires — no exit logic, no fair value signal needed.

    Parameters
    ----------
    positions : dict[str, int]
        {product: target_signed_position}
        Positive = long (buy to +limit), negative = short (sell to -limit).
        Use the position limit as the magnitude, e.g. +10 or -10.

    Example
    -------
    BuyNHold({
        'MICROCHIP_OVAL':    -10,
        'MICROCHIP_TRIANGLE':-10,
        'PANEL_1X2':         +10,
        'SLEEP_POD_NYLON':   +10,
    })
    """

    def __init__(self, positions: dict):
        super().__init__("BuyNHold", list(positions.keys()))
        self.positions = positions  # {product: target_signed_position}

    def generate(self, state: TradingState,
                 implied_positions: dict,
                 order_budgets: dict,
                 arb_positions: dict,
                 resting_book: dict) -> dict:

        result = {p: [] for p in self.products}

        for product, target in self.positions.items():
            current = implied_positions.get(product, 0)
            gap = target - current

            if gap == 0:
                continue  # already at target

            direction = 1 if gap > 0 else -1

            result[product] += self.taker_execution(
                product,
                fair_value=999999 if direction > 0 else 0,
                buy_threshold=-10 if direction > 0 else 999,
                sell_threshold=999 if direction > 0 else -10,
                order_budgets=order_budgets,
                implied_positions=implied_positions,
                resting_book=resting_book,
                max_quantity=abs(gap),
            )

        return result



class MarketMake(MultiProductStrategy):

    def __init__(self, products):
        super().__init__("MarketMake", products)
        self.products = products


    def generate(self, state: TradingState,
                 implied_positions: dict,
                 order_budgets: dict,
                 arb_positions: dict,
                 resting_book: dict) -> dict:

        result = {p: [] for p in self.products}

        return result

# Single Product Strategies

class MM1(SingleProductStrategy):
    def __init__(self):
        super().__init__("SNACKPACK_RASPBERRY", 10)

    def fair_value(self, state: TradingState):
        if true_mid(self.symbol, state) != -1:
            return true_mid(self.symbol, state)
        else:
            return state.traderData[self.symbol]

class MM2(SingleProductStrategy):
    def __init__(self):
        super().__init__("OXYGEN_SHAKE_EVENING_BREATH", 10)

    def fair_value(self, state: TradingState):
        if true_mid(self.symbol, state) != -1:
            return true_mid(self.symbol, state)
        else:
            return state.traderData[self.symbol]


class MM3(SingleProductStrategy):
    def __init__(self):
        super().__init__("SNACKPACK_STRAWBERRY", 10)

    def fair_value(self, state: TradingState):
        if true_mid(self.symbol, state) != -1:
            return true_mid(self.symbol, state)
        else:
            return state.traderData[self.symbol]


class MM4(SingleProductStrategy):
    def __init__(self):
        super().__init__("OXYGEN_SHAKE_MORNING_BREATH", 10)

    def fair_value(self, state: TradingState):
        if true_mid(self.symbol, state) != -1:
            return true_mid(self.symbol, state)
        else:
            return state.traderData[self.symbol]


class MM5(SingleProductStrategy):
    def __init__(self):
        super().__init__("TRANSLATOR_VOID_BLUE", 10)

    def fair_value(self, state: TradingState):
        if true_mid(self.symbol, state) != -1:
            return true_mid(self.symbol, state)
        else:
            return state.traderData[self.symbol]


class MM6(SingleProductStrategy):
    def __init__(self):
        super().__init__("UV_VISOR_ORANGE", 10)

    def fair_value(self, state: TradingState):
        if true_mid(self.symbol, state) != -1:
            return true_mid(self.symbol, state)
        else:
            return state.traderData[self.symbol]


class MM7(SingleProductStrategy):
    def __init__(self):
        super().__init__("UV_VISOR_RED", 10)

    def fair_value(self, state: TradingState):
        if true_mid(self.symbol, state) != -1:
            return true_mid(self.symbol, state)
        else:
            return state.traderData[self.symbol]


class MM8(SingleProductStrategy):
    def __init__(self):
        super().__init__("TRANSLATOR_ASTRO_BLACK", 10)

    def fair_value(self, state: TradingState):
        if true_mid(self.symbol, state) != -1:
            return true_mid(self.symbol, state)
        else:
            return state.traderData[self.symbol]


class Trader:

    def bid(self):
        return 15

    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""
        # get new trader data
        trader_data = jsonpickle.decode(state.traderData) if state.traderData else {
            "debug": {
            },
            "arb_positions": {},
            "SNACKPACK_RASPBERRY": 10000,
            "SNACKPACK_STRAWBERRY": 10000,
            "TRANSLATOR_VOID_BLUE": 10000,
            "OXYGEN_SHAKE_EVENING_BREATH": 10000,
            "OXYGEN_SHAKE_MORNING_BREATH": 10000,
            "UV_VISOR_ORANGE": 10000,
            "UV_VISOR_RED": 10000,
            "TRANSLATOR_ASTRO_BLACK": 10000,

        }
        mm_symbols = ["SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY", "TRANSLATOR_VOID_BLUE", "OXYGEN_SHAKE_EVENING_BREATH",
                      "OXYGEN_SHAKE_MORNING_BREATH", "UV_VISOR_ORANGE", "UV_VISOR_RED", "TRANSLATOR_ASTRO_BLACK"]
        for symbol in mm_symbols:
            if true_mid(symbol, state) != -1:
                trader_data[symbol] = true_mid(symbol, state)
        # make sure traderdata is up to date
        state.traderData = trader_data



        # set up position tracking
        pos_limits = {
            "GALAXY_SOUNDS_DARK_MATTER": 10,
            "GALAXY_SOUNDS_BLACK_HOLES": 10,
            "GALAXY_SOUNDS_PLANETARY_RINGS": 10,
            "GALAXY_SOUNDS_SOLAR_WINDS": 10,
            "GALAXY_SOUNDS_SOLAR_FLAMES": 10,
            "SLEEP_POD_SUEDE": 10,
            "SLEEP_POD_LAMB_WOOL": 10,
            "SLEEP_POD_POLYESTER": 10,
            "SLEEP_POD_NYLON": 10,
            "SLEEP_POD_COTTON": 10,
            "MICROCHIP_CIRCLE": 10,
            "MICROCHIP_OVAL": 10,
            "MICROCHIP_SQUARE": 10,
            "MICROCHIP_RECTANGLE": 10,
            "MICROCHIP_TRIANGLE": 10,
            "PEBBLES_XS": 10,
            "PEBBLES_S": 10,
            "PEBBLES_M": 10,
            "PEBBLES_L": 10,
            "PEBBLES_XL": 10,
            "ROBOT_VACUUMING": 10,
            "ROBOT_MOPPING": 10,
            "ROBOT_DISHES": 10,
            "ROBOT_LAUNDRY": 10,
            "ROBOT_IRONING": 10,
            "UV_VISOR_YELLOW": 10,
            "UV_VISOR_AMBER": 10,
            "UV_VISOR_ORANGE": 10,
            "UV_VISOR_RED": 10,
            "UV_VISOR_MAGENTA": 10,
            "TRANSLATOR_SPACE_GRAY": 10,
            "TRANSLATOR_ASTRO_BLACK": 10,
            "TRANSLATOR_ECLIPSE_CHARCOAL": 10,
            "TRANSLATOR_GRAPHITE_MIST": 10,
            "TRANSLATOR_VOID_BLUE": 10,
            "PANEL_1X2": 10,
            "PANEL_2X2": 10,
            "PANEL_1X4": 10,
            "PANEL_2X4": 10,
            "PANEL_4X4": 10,
            "OXYGEN_SHAKE_MORNING_BREATH": 10,
            "OXYGEN_SHAKE_EVENING_BREATH": 10,
            "OXYGEN_SHAKE_MINT": 10,
            "OXYGEN_SHAKE_CHOCOLATE": 10,
            "OXYGEN_SHAKE_GARLIC": 10,
            "SNACKPACK_CHOCOLATE": 10,
            "SNACKPACK_VANILLA": 10,
            "SNACKPACK_PISTACHIO": 10,
            "SNACKPACK_STRAWBERRY": 10,
            "SNACKPACK_RASPBERRY": 10,
        }

        raw_positions = {}
        order_budgets = {}
        resting_book = {}
        for product in pos_limits:
            # get positions and calculate order budgets
            raw_positions[product] = state.position.get(product, 0)
            order_budgets[product] = {
                'buy': pos_limits[product] - raw_positions[product],
                'sell': pos_limits[product] + raw_positions[product]
            }

            order_depth = state.order_depths.get(product, -1)
            # create a mutable resting book
            if order_depth != -1:
                resting_book[product] = {
                    "bids": dict(order_depth.buy_orders),
                    "asks": dict(order_depth.sell_orders),
                }
            else:
                resting_book[product] = {"bids": {}, "asks": {}}

        implied_positions = dict(raw_positions)

        BUY_N_HOLD = {
            # ─── TIER A: HIGH CONFIDENCE (all 3 signals agree) ───────────────
            'MICROCHIP_OVAL': -10,  # Δ34=-2315, slope=-139
            'ROBOT_LAUNDRY': -10,  # Δ34=-1135, slope=-20
            'ROBOT_VACUUMING': -10,  # Δ34=-497,  slope=-430
            'ROBOT_IRONING': -10,  # Δ34=-1035, slope=+28 (mild)
            'ROBOT_DISHES': +10,  # Δ34=+694,  slope=+252
            'ROBOT_MOPPING': +10,  # Δ34=+739,  slope=+313
            'OXYGEN_SHAKE_GARLIC': +10,  # Δ34=+1103, slope=+366
            'PANEL_2X4': +10,  # Δ34=+669,  slope=+80
            'PEBBLES_XS': -10,  # Δ34=-912,  slope=-75 (decel)
            'TRANSLATOR_GRAPHITE_MIST': +10,  # Δ34=+511,  slope=+6  (NEW)
            'SNACKPACK_CHOCOLATE': -10,  # Δ34=-232,  slope=+71 (small)
            'SNACKPACK_PISTACHIO': -10,  # Δ34=-143,  slope=-117 (NEW, small)

            # ─── TIER B: MEDIUM CONFIDENCE (recent direction matches) ────────
            'MICROCHIP_TRIANGLE': -10,  # Δ34=-1615, slope=-246
            'PANEL_1X2': +10,  # Δ34=+975,  slope=+147
            'SLEEP_POD_NYLON': +10,  # Δ34=+949,  slope=+61
            'OXYGEN_SHAKE_CHOCOLATE': +10,  # Δ34=+828,  slope=+129
            'GALAXY_SOUNDS_PLANETARY_RINGS': -10,  # Δ34=-944,  slope=-571 (NEW)
            'SLEEP_POD_LAMB_WOOL': +10,  # Δ34=+383,  slope=+10  (NEW)
            'GALAXY_SOUNDS_SOLAR_FLAMES': -10,  # Δ34=-338,  slope=-48  (NEW)
        }
        # create strategy objects
        strategies = [
            BuyNHold(BUY_N_HOLD),
            MM1(),
            MM2(),
            MM3(),
            MM4(),
            MM5(),
            MM6(),
            MM7(),
            MM8(),
        ]

        # Orders to be placed on exchange matching engine
        result = {}

        for strategy in strategies:
            orders = strategy.generate(state, implied_positions, order_budgets,
                                                        trader_data['arb_positions'], resting_book)
            if isinstance(orders, dict):
                for product, product_orders in orders.items():
                    result.setdefault(product, []).extend(product_orders)
            else:
                result.setdefault(strategy.symbol, []).extend(orders)

        # print(trader_data)
        trader_data_encode = jsonpickle.encode(trader_data)

        conversions = 0
        return result, conversions, trader_data_encode
