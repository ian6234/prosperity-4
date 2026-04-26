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


class VelvetArb(MultiProductStrategy):
    """
    Combined VEV OU mean-reversion take + options delta amplifier.

    Structure
    ---------
    Layer 1 — VEV OU take
        VELVETFRUIT_EXTRACT is an OU process: mu ≈ 5251, sigma_s ≈ 16, half-life ≈ 279 ticks.
        Enter VEV_SIZE units in the reversion direction when |n_sigma| >= ENTRY_SIGMA (1.5).
        Exit all positions when price crosses back through mu.

    Layer 2 — Options amplifier
        On the same entry signal, take OPT_QTY units of every strike in OPTION_STRIKES.
        All legs exit together when VEV crosses mu.
        Entry BS price of each strike is stored in traderData['VELVETFRUIT_EXTRACT']['entry_prices']
        as a dict keyed by strike int, so adding/removing strikes requires no schema changes.
        Options are skipped entirely when TTE < TTE_MIN to avoid end-of-round theta decay.

    Adding or removing a strike: edit OPTION_STRIKES only — nothing else changes.

    Key numbers (calibrated on 3-day backtest, all six strikes)
        sigma_s = 16 → 1.5σ entry at VEV ≈ 5227 / 5275
        Effective delta at entry: ~200 (VEV) + ~480 (options) = ~680 total
        Simulated net: ~96k/day vs ~32k/day VEV-only

    traderData schema (under key 'VELVETFRUIT_EXTRACT')
        last_mid:     float             — last observed VEV mid, fallback if book empty
        ou_tier:      int               — trade direction: -1 short / 0 flat / +1 long
        entry_prices: dict[int, float]  — {strike: BS price at entry} for open options;
                                          empty dict when flat
    """

    VEV = 'VELVETFRUIT_EXTRACT'

    # ── OU parameters ─────────────────────────────────────────────────────────
    OU_MU: float = 5251.0
    OU_SIGMA: float = 16.0
    ENTRY_SIGMA: float = 1.5

    # ── Sizes ─────────────────────────────────────────────────────────────────
    VEV_SIZE: int = 200
    OPT_QTY: int = 300  # units per strike (capped by position limit)

    # ── Option strikes to amplify with ────────────────────────────────────────
    # Each entry: (strike: int, position_limit: int)
    # Remove a row to drop a strike; add a row to include a new one.
    OPTION_STRIKES: list = [
        (4000, 300),
        (4500, 300),
        (5000, 300),
        (5100, 300),
        (5200, 300),
        (5300, 300),
        (5400, 300),
        (5500, 300),
    ]

    # ── TTE guard ─────────────────────────────────────────────────────────────
    ROUND_TTE: int = 5
    TICKS_PER_ROUND: int = 10000
    TTE_MIN: float = 0.5

    # ── Black-Scholes ──────────────────────────────────────────────────────────
    IV: float = 0.0146

    def __init__(self):
        option_products = [f'VEV_{K}' for K, _ in self.OPTION_STRIKES]
        super().__init__("VelvetArb", [self.VEV] + option_products)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _norm_cdf(self, x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def _bs_price(self, S: float, K: int, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self._norm_cdf(d1) - K * self._norm_cdf(d2)

    def _tte(self, timestamp: int) -> float:
        return self.ROUND_TTE - timestamp / (self.TICKS_PER_ROUND * 100)

    def _mid(self, product: str, state: TradingState, fallback: float) -> float:
        od = state.order_depths.get(product)
        if od and od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return fallback

    def _take(self, product: str, direction: int, qty: int,
              order_budgets: dict, implied_positions: dict,
              resting_book: dict, result: dict):
        """Take `qty` units of `product` in `direction` (+1 buy / -1 sell)."""
        if qty <= 0:
            return
        if direction > 0:
            result[product] += self.taker_execution(
                product, self.OU_MU * 2,  # wide FV → threshold=-10 always fires for buys
                buy_threshold=-10, sell_threshold=999,
                order_budgets=order_budgets,
                implied_positions=implied_positions,
                resting_book=resting_book,
                max_quantity=qty,
            )
        else:
            result[product] += self.taker_execution(
                product, 0,  # wide FV → threshold=-10 always fires for sells
                buy_threshold=999, sell_threshold=-10,
                order_budgets=order_budgets,
                implied_positions=implied_positions,
                resting_book=resting_book,
                max_quantity=qty,
            )

    # ── Core generate ──────────────────────────────────────────────────────────

    def generate(self, state: TradingState,
                 implied_positions: dict,
                 order_budgets: dict,
                 arb_positions: dict,
                 resting_book: dict) -> dict:

        result = {p: [] for p in self.products}
        vd = state.traderData[self.VEV]

        S = self._mid(self.VEV, state, vd['last_mid'])
        T = self._tte(state.timestamp)
        ns = (self.OU_MU - S) / self.OU_SIGMA

        current_tier = vd.get('ou_tier', 0)

        # ── State transition ──────────────────────────────────────────────────
        if current_tier == 0:
            if ns >= self.ENTRY_SIGMA:
                new_tier = 1
            elif ns <= -self.ENTRY_SIGMA:
                new_tier = -1
            else:
                new_tier = 0
        else:
            exiting = (current_tier == 1 and ns < 0) or \
                      (current_tier == -1 and ns > 0)
            new_tier = 0 if exiting else current_tier

        vd['ou_tier'] = new_tier

        # ── Open ──────────────────────────────────────────────────────────────
        if new_tier != 0:
            direction = new_tier

            # Layer 1: VEV — use OU_MU as FV so threshold=-10 fires whenever below/above mu
            vev_gap = self.VEV_SIZE * direction - implied_positions.get(self.VEV, 0)
            result[self.VEV] += self.taker_execution(
                self.VEV, self.OU_MU,
                buy_threshold=-10 if direction > 0 else 999,
                sell_threshold=-10 if direction < 0 else 999,
                order_budgets=order_budgets,
                implied_positions=implied_positions,
                resting_book=resting_book,
                max_quantity=abs(vev_gap),
            )

            # Layer 2: options — iterate over every configured strike
            entry_prices = {}
            if T >= self.TTE_MIN:
                for K, pos_limit in self.OPTION_STRIKES:
                    product = f'VEV_{K}'
                    entry_prices[K] = self._bs_price(S, K, T, self.IV)
                    qty = min(self.OPT_QTY, pos_limit)
                    opt_gap = qty * direction - implied_positions.get(product, 0)
                    self._take(product, direction, abs(opt_gap),
                               order_budgets, implied_positions, resting_book, result)

            # Persist entry prices (empty dict if TTE guard prevented opening)
            vd['entry_prices'] = entry_prices

        # ── Close ─────────────────────────────────────────────────────────────
        elif new_tier == 0:
            direction = current_tier

            # Layer 1: close VEV
            vev_pos = implied_positions.get(self.VEV, 0)
            result[self.VEV] += self.taker_execution(
                self.VEV, self.OU_MU,
                buy_threshold=-10 if vev_pos < 0 else 999,
                sell_threshold=-10 if vev_pos > 0 else 999,
                order_budgets=order_budgets,
                implied_positions=implied_positions,
                resting_book=resting_book,
                max_quantity=abs(vev_pos),
            )

            # Layer 2: close every option that was opened this trade
            entry_prices = vd.get('entry_prices', {})
            for K, _ in self.OPTION_STRIKES:
                if K not in entry_prices:
                    continue  # not opened (TTE guard, or strike added after entry)
                product = f'VEV_{K}'
                opt_pos = implied_positions.get(product, 0)
                if opt_pos == 0:
                    continue
                self._take(product, -direction, abs(opt_pos),
                           order_budgets, implied_positions, resting_book, result)

            vd['entry_prices'] = {}  # clear on close

        return result


# Single Product Strategies


class HydrogelPacksOU(SingleProductStrategy):
    """
    Two-layer strategy for HYDROGEL_PACK.

    Layer 1 — OU Mean-Reversion TAKE (primary alpha source)
    --------------------------------------------------------
    HYDROGEL_PACK is an OU process with:
        mu      = 9991   (long-run mean, stable across rounds)
        sigma_s = 32     (stationary std, empirical range ~9891–10079)
        half-life ~ 300 ticks

    We size positions in two tiers using HYSTERESIS to avoid boundary churn:
        Tier 1: enter when |n_sigma| >= TIER1_ENTER (1.0), exit at TIER1_EXIT (0.0 i.e. mu)
        Tier 2: enter when |n_sigma| >= TIER2_ENTER (2.0), step back to Tier 1 at TIER2_EXIT (1.0)

    The current tier is persisted in traderData['HYDROGEL_PACK']['ou_tier'] so
    hysteresis survives across timestamps (Lambda statelessness).

    Layer 2 — Deep-Mid Market-Making (background income)
    -----------------------------------------------------
    Fair value = (bid_price_2 + ask_price_2) / 2  — the stable structural-bot anchor.
    When the L1 spread narrows to 7 or 9 (a noise bot hit one side), we lean
    FV by ±REGIME_LEAN ticks to ride the micro-trend for one tick.

    Order budget is consumed by the OU take first, so MM quotes are naturally
    smaller when we're heavily positioned — no double-counting needed.
    """

    # ── OU parameters ────────────────────────────────────────────────────────
    OU_MU: float    = 9991.0   # long-run mean (hardcode; recalibrate each round)
    OU_SIGMA: float = 32.0     # stationary std

    # ── Tier entry / exit thresholds (in units of OU_SIGMA) ──────────────────
    TIER1_ENTER: float = 1.0   # enter Tier 1 at 1-sigma  (~9959 / ~10023)
    TIER1_EXIT:  float = 0.0   # exit Tier 1 when price returns past mu
    TIER2_ENTER: float = 2.0   # enter Tier 2 at 2-sigma  (~9927 / ~10055)
    TIER2_EXIT:  float = 1.0   # step back to Tier 1 when |n_sigma| < 1

    TIER1_SIZE: int = 100      # units held in Tier 1
    TIER2_SIZE: int = 200      # units held in Tier 2 (= position limit)

    # ── Spread-regime lean ───────────────────────────────────────────────────
    REGIME_LEAN: float = 4.0   # ticks added to FV when spread signals direction

    def __init__(self, symbol: str, pos_limit: int):
        super().__init__(symbol, pos_limit)
        self.max_threshold = 1  # used by base market_make(); keep tight

    # ─────────────────────────────────────────────────────────────────────────
    # Fair value for market-making (deep mid + optional regime lean)
    # ─────────────────────────────────────────────────────────────────────────
    def fair_value(self, state: TradingState) -> float:
        od = state.order_depths.get(self.symbol)
        if od is None:
            return float(state.traderData['HYDROGEL_PACK']['last_mid'])

        bids = sorted(od.buy_orders.keys(), reverse=True)
        asks = sorted(od.sell_orders.keys())

        if len(bids) < 2 or len(asks) < 2:
            return float(state.traderData['HYDROGEL_PACK']['last_mid'])

        ask1, ask2 = asks[0], asks[1]
        bid1, bid2 = bids[0], bids[1]
        spread      = ask1 - bid1
        deep_mid    = (bid2 + ask2) / 2.0

        # Spread-regime signal: a noise bot just crossed one side
        if spread == 7:
            # Ask side tightened → someone bought aggressively → lean bullish
            return deep_mid + self.REGIME_LEAN
        if spread == 9:
            # Bid side tightened → someone sold aggressively → lean bearish
            return deep_mid - self.REGIME_LEAN

        return deep_mid

    # ─────────────────────────────────────────────────────────────────────────
    # OU tier transition (with hysteresis)
    # Returns the desired signed position size based on current n_sigma and
    # the previously stored tier (to implement hysteresis correctly).
    # ─────────────────────────────────────────────────────────────────────────
    def _ou_desired_position(self, mid: float, current_tier: int) -> tuple:
        """
        Returns (new_tier, target_position).
        current_tier: -2, -1, 0, +1, +2  (persisted in traderData)
        """
        dev   = self.OU_MU - mid          # positive → below mu → want long
        n_sig = dev / self.OU_SIGMA

        # --- transition logic with hysteresis ---
        if current_tier == 0:
            if   n_sig >=  self.TIER2_ENTER:  new_tier =  2
            elif n_sig >=  self.TIER1_ENTER:  new_tier =  1
            elif n_sig <= -self.TIER2_ENTER:  new_tier = -2
            elif n_sig <= -self.TIER1_ENTER:  new_tier = -1
            else:                             new_tier =  0

        elif current_tier == 1:
            if   n_sig >=  self.TIER2_ENTER:  new_tier =  2
            elif n_sig <   self.TIER1_EXIT:   new_tier =  0   # price crossed mu → flat
            else:                             new_tier =  1

        elif current_tier == 2:
            if   n_sig <   self.TIER2_EXIT:                    # stepped back inside 1-sigma
                new_tier = 1 if n_sig >= self.TIER1_EXIT else 0
            else:                             new_tier =  2

        elif current_tier == -1:
            if   n_sig <= -self.TIER2_ENTER:  new_tier = -2
            elif n_sig >  -self.TIER1_EXIT:   new_tier =  0
            else:                             new_tier = -1

        elif current_tier == -2:
            if   n_sig >  -self.TIER2_EXIT:
                new_tier = -1 if n_sig <= -self.TIER1_EXIT else 0
            else:                             new_tier = -2

        else:
            new_tier = 0  # safety

        target_map = {0: 0, 1: self.TIER1_SIZE, 2: self.TIER2_SIZE,
                      -1: -self.TIER1_SIZE, -2: -self.TIER2_SIZE}
        return new_tier, target_map[new_tier]

    # ─────────────────────────────────────────────────────────────────────────
    # Main generate
    # ─────────────────────────────────────────────────────────────────────────
    def generate(self, state: TradingState,
                 implied_positions: dict,
                 order_budgets: dict,
                 arb_positions: dict,
                 resting_book: dict):

        product = self.symbol
        hp_data  = state.traderData['HYDROGEL_PACK']

        # current position and persisted OU tier
        current_pos  = implied_positions.get(product, 0)
        current_tier = hp_data.get('ou_tier', 0)

        # current mid (top-of-book) for OU signal
        od = state.order_depths.get(product)
        if od and od.buy_orders and od.sell_orders:
            mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        else:
            mid = float(hp_data['last_mid'])

        # ── Layer 1: OU take ─────────────────────────────────────────────────
        new_tier, ou_target = self._ou_desired_position(mid, current_tier)

        # Persist updated tier immediately so next tick uses correct hysteresis
        hp_data['ou_tier'] = new_tier

        ou_gap = ou_target - current_pos   # positive → need to buy, negative → need to sell

        take_orders = []
        if ou_gap > 0:
            # Cross the ~8-tick half-spread to get long.
            # buy_threshold = -10: buy if best_ask + (-10) <= OU_MU, i.e. ask <= OU_MU + 10
            # Fires whenever we're below mu (which is exactly when we want to buy).
            take_orders = self.taker_execution(
                product, self.OU_MU,
                buy_threshold=-10,
                sell_threshold=999,    # never sell in this pass
                order_budgets=order_budgets,
                implied_positions=implied_positions,
                resting_book=resting_book,
                max_quantity=ou_gap,
            )
        elif ou_gap < 0:
            # sell_threshold = -10: sell if best_bid - (-10) >= OU_MU, i.e. bid >= OU_MU - 10
            # Fires whenever we're above mu (which is exactly when we want to sell).
            take_orders = self.taker_execution(
                product, self.OU_MU,
                buy_threshold=999,
                sell_threshold=-10,
                order_budgets=order_budgets,
                implied_positions=implied_positions,
                resting_book=resting_book,
                max_quantity=abs(ou_gap),
            )

        # ── Layer 2: deep-mid market-making ──────────────────────────────────
        # fair_value() uses deep mid ± regime lean.
        # order_budgets is already reduced by the OU take above, so MM size
        # is automatically smaller when we're heavily positioned.
        fv = self.fair_value(state)
        mm_orders = self.market_make(fv, order_budgets, resting_book)

        return take_orders + mm_orders


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
            VelvetArb(),
            HydrogelPacksOU("HYDROGEL_PACK", 200)
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
