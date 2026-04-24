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
                        resting_book):

        position = implied_positions[product]

        orders = []

        remaining_bids = resting_book[product]['bids']
        remaining_asks = resting_book[product]['asks']

        last_ask = -1
        last_bid = -1

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

class VelvetOptions(MultiProductStrategy):

    STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
    # fit on close to money options
    STRIKES_FIT = [5000, 5100, 5200, 5300, 5400, 5500]
    ROUND_TTE = 6  # 8 for testing, 5 for round 3, 4 for round 4, ...
    TICKS_PER_ROUND = 10000
    MIN_IV = 0.001
    TAKER_THRESHOLD = 2

    # parameters for hedging aggression
    HEDGE_THRESHOLD = 15
    URGENT_THRESHOLD = 50

    def __init__(self):
        super().__init__("VelvetOptions", ['VELVETFRUIT_EXTRACT'] + [f'VEV_{k}' for k in self.STRIKES])


    # ---------------------BS FUNCTIONS-----------------------------------------

    # Black-Scholes for VEV trading (TIME==Days (timesteps/10000), VOL==Daily std dev 100*std dev)
    def norm_pdf(self, x):
        return (1 / (math.sqrt(2 * math.pi))) * math.exp(-0.5 * x ** 2)

    def phi(self, x):
        # 'Cumulative distribution function for the standard normal distribution'
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def d1(self, S, strike, time, r, sigma):
        return (math.log(S / strike) + (r + (sigma ** 2) / 2) * time) / (sigma * math.sqrt(time))

    def d2(self, S, strike, time, r, sigma):
        return self.d1(S, strike, time, r, sigma) - (sigma * math.sqrt(time))

    def call_vega(self, S, strike, time, r, sigma):
        return S * math.sqrt(time) * self.norm_pdf(self.d1(S, strike, time, r, sigma))

    def call_price(self, S, strike, time, r, sigma):
        return S * self.phi(self.d1(S, strike, time, r, sigma)) - strike * math.exp(
            -r * time) * self.phi(self.d2(S, strike, time, r, sigma))

    def call_implied_vol(self, S, strike, time, r, opt_price, sigma_est, iterations):

        # guard: if price is at or below intrinsic, IV is undefined
        intrinsic = max(S - strike, 0.0)
        if opt_price <= intrinsic:
            return -1

        for i in range(iterations):
            diff = self.call_price(S, strike, time, r, sigma_est) - opt_price
            try:
                sigma_est -= diff / self.call_vega(S, strike, time, r, sigma_est)
            except:
                return -1
            if abs(diff) < 0.001:
                break
            else:
                continue
        return sigma_est

    def call_delta(self, S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        return self.phi(self.d1(S, K, T, r, sigma))

    def m_t(self, strike, underlying, tte):
        return np.log(strike / underlying) / np.sqrt(tte)

    # Fit m_t, v_t
    def fit_quadratic(self, x, y):
        return np.polyfit(x, y, 2)

    #----------------------------------------------------------------------

    def get_expiry(self, timestamp):
        return self.ROUND_TTE - (timestamp / (self.TICKS_PER_ROUND * 100))

    def fit_surface(self, S, T, state):
        moneyness = []
        ivs = []

        for strike in self.STRIKES_FIT:
            product = f'VEV_{strike}'
            mid = true_mid(product, state)
            if mid <= 0:
                continue

            # skip deep OTM where price is near zero
            intrinsic = max(S - strike, 0)
            if mid < 0.5 and intrinsic < 0.5:
                continue

            iv = self.call_implied_vol(S, strike, T, 0, mid, sigma_est=0.02, iterations=100)
            if iv < self.MIN_IV or iv == -1:
                continue

            moneyness.append(self.m_t(strike, S, T))
            ivs.append(iv)

        if len(moneyness) < 3:
            return None  # not enough points for quadratic

        return self.fit_quadratic(moneyness, ivs)

    # calculates the fair value for one option. returns the fair price, and the fitted iv (for delta calc later)
    def option_fair_value(self, strike, S, T, coeffs):
        mt = self.m_t(strike, S, T)
        fitted_iv = np.polyval(coeffs, mt)
        fitted_iv = max(fitted_iv, self.MIN_IV)  # floor IV
        return self.call_price(S, strike, T, 0, fitted_iv), fitted_iv

    # runs taker logic on every option
    def generate_option_orders(self, S, T, coeffs, state,
                               implied_positions, order_budgets, resting_book):
        result = {}
        taker_candidates = []

        # fitted strikes only - compute deviations
        for strike in self.STRIKES_FIT:
            product = f'VEV_{strike}'
            mid = true_mid(product, state)
            if mid <= 0:
                result[product] = []
                continue

            fv, fitted_iv = self.option_fair_value(strike, S, T, coeffs)
            price_dev = mid - fv
            taker_candidates.append((abs(price_dev), strike, fv, fitted_iv, price_dev))
            result[product] = []

        # sort by abs deviation descending
        taker_candidates.sort(key=lambda x: x[0], reverse=True)

        # taker pass only
        for abs_dev, strike, fv, fitted_iv, price_dev in taker_candidates:
            product = f'VEV_{strike}'
            buy_thresh = self.TAKER_THRESHOLD if price_dev < 0 else 999
            sell_thresh = self.TAKER_THRESHOLD if price_dev > 0 else 999
            result[product].extend(
                self.taker_execution(product, fv, buy_thresh, sell_thresh,
                                     order_budgets, implied_positions, resting_book)
            )

        return result

    # delta calculator
    def calculate_net_delta(self, S, T, coeffs, implied_positions):
        net_delta = 0.0
        for strike in self.STRIKES_FIT:
            product = f'VEV_{strike}'
            position = implied_positions.get(product, 0)
            if position == 0:
                continue
            _, fitted_iv = self.option_fair_value(strike, S, T, coeffs)
            delta = self.call_delta(S, strike, T, 0, fitted_iv)
            net_delta += position * delta
        # VEV itself has delta = 1 per unit
        net_delta += implied_positions.get('VELVETFRUIT_EXTRACT', 0)
        return net_delta

    # creates orders to hedge the current net delta exposure.
    def hedge_orders(self, net_delta, S, implied_positions, order_budgets, resting_book):
        orders = []
        if abs(net_delta) < self.HEDGE_THRESHOLD:
            return orders

        need_to_sell = net_delta > 0

        if abs(net_delta) >= self.URGENT_THRESHOLD:
            # cross the spread aggressively - pass wide thresholds
            buy_thresh = -10 if not need_to_sell else 999
            sell_thresh = -10 if need_to_sell else 999
        else:
            # gentle - only fill at best available (threshold=0 means fill at mid or better)
            buy_thresh = 0 if not need_to_sell else 999
            sell_thresh = 0 if need_to_sell else 999

        orders.extend(
            self.taker_execution(
                'VELVETFRUIT_EXTRACT', S,
                buy_thresh, sell_thresh,
                order_budgets, implied_positions, resting_book
            )
        )
        return orders


    def generate(self, state: TradingState,
                 implied_positions: dict,
                 order_budgets: dict,
                 arb_positions: dict,
                 resting_book: dict):

        result = {product: [] for product in self.products}

        # 1. VEV spot
        S = true_mid('VELVETFRUIT_EXTRACT', state)
        if S <= 0:
            S = float(state.traderData['VELVETFRUIT_EXTRACT']['last_mid'])

        # 2. TTE
        T = self.get_expiry(state.timestamp)
        if T <= 0:
            return result

        # 3. Fit surface
        coeffs = self.fit_surface(S, T, state)
        if coeffs is None:
            return result

        # DEBUG - log surface fit and deviations for first 5 ticks
        if state.timestamp < 500:
            debug = {
                'timestamp': state.timestamp,
                'S': S,
                'T': T,
                'coeffs': list(coeffs),
                'strikes': {}
            }
            for strike in self.STRIKES_FIT:
                product = f'VEV_{strike}'
                mid = true_mid(product, state)
                if mid > 0:
                    fv, fitted_iv = self.option_fair_value(strike, S, T, coeffs)
                    debug['strikes'][strike] = {
                        'mid': mid,
                        'fv': round(fv, 3),
                        'iv': round(fitted_iv, 5),
                        'dev': round(mid - fv, 3)
                    }
            state.traderData['debug']['options'] = debug

        # 4. Option taker orders
        option_orders = self.generate_option_orders(
            S, T, coeffs, state,
            implied_positions, order_budgets, resting_book
        )
        for product, orders in option_orders.items():
            result[product].extend(orders)

        # 5. Delta hedge
        net_delta = self.calculate_net_delta(S, T, coeffs, implied_positions)
        result['VELVETFRUIT_EXTRACT'].extend(
            self.hedge_orders(net_delta, S, implied_positions, order_budgets, resting_book)
        )

        return result



# Single Product Strategies

# keeping this here for reference
class HydrogelPacks(SingleProductStrategy):

    def __init__(self, symbol: str, pos_limit: int):
        super().__init__(symbol, pos_limit)
        self.max_threshold = 1

    # AR(1) fair value
    def fair_value(self, state: TradingState) -> float:
        current_price = true_mid(self.symbol, state)
        if current_price == -1:
            return float(state.traderData['HYDROGEL_PACK']['last_mid'])
        return current_price


class Trader:

    def bid(self):
        return 15

    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""
        # get new trader data
        trader_data = jsonpickle.decode(state.traderData) if state.traderData else {
            "HYDROGEL_PACK": {
                'last_mid': 10000,
            },
            "VELVETFRUIT_EXTRACT": {
                'last_mid': 5250,
            },
            "debug": {
                "options": 0,
            },
            "arb_positions": {}
        }
        # track last hydrogel pack mid-price so we don't lose it
        mid = true_mid('HYDROGEL_PACK', state)
        if mid != -1:
            trader_data['HYDROGEL_PACK']['last_mid'] = mid

        # track last velvet mid-price so we don't lose it
        mid = true_mid('VELVETFRUIT_EXTRACT', state)
        if mid != -1:
            trader_data['VELVETFRUIT_EXTRACT']['last_mid'] = mid

        # make sure traderdata is up to date
        state.traderData = trader_data

        # set up position tracking
        pos_limits = {"HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200,
                       "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
                       "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300, "VEV_6000": 300, "VEV_6500": 300 }
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

        # create strategy objects
        strategies = [
            VelvetOptions(),
            HydrogelPacks("HYDROGEL_PACK", 200)
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

        print(trader_data)
        trader_data_encode = jsonpickle.encode(trader_data)

        conversions = 0
        return result, conversions, trader_data_encode
