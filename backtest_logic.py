from datamodel import *
from algo_test import Trader
import numpy as np
import pandas as pd


class OrderBook:
    """
    Maintains resting maker orders for a single symbol, split into market
    and algo (SUBMISSION) quantity at each price level.

    Conventions (matching OrderDepth):
      buy_orders  (bids): price -> +quantity
      sell_orders (asks): price -> -quantity

    Market resting quantity has time priority over algo resting quantity,
    so external incoming orders consume market quantity at a level first.
    """

    def __init__(self, symbol: Symbol):
        self.symbol = symbol
        # market resting orders (from price CSV)
        self.mkt_buy_orders: Dict[int, int] = {}   # price -> +qty
        self.mkt_sell_orders: Dict[int, int] = {}  # price -> -qty
        # algo's resting orders (unfilled maker orders from step 3)
        self.my_buy_orders: Dict[int, int] = {}
        self.my_sell_orders: Dict[int, int] = {}

    @property
    def buy_orders(self) -> Dict[int, int]:
        combined = dict(self.mkt_buy_orders)
        for p, q in self.my_buy_orders.items():
            combined[p] = combined.get(p, 0) + q
        return combined

    @property
    def sell_orders(self) -> Dict[int, int]:
        combined = dict(self.mkt_sell_orders)
        for p, q in self.my_sell_orders.items():
            combined[p] = combined.get(p, 0) + q
        return combined

    # ------------------------------------------------------------------
    def add_to_book(self, order: Order, is_mine: bool = False) -> None:
        """Add a resting maker order to the book."""
        buy_side = self.my_buy_orders if is_mine else self.mkt_buy_orders
        sell_side = self.my_sell_orders if is_mine else self.mkt_sell_orders
        if order.quantity > 0:
            buy_side[order.price] = buy_side.get(order.price, 0) + order.quantity
        elif order.quantity < 0:
            sell_side[order.price] = sell_side.get(order.price, 0) + order.quantity

    # ------------------------------------------------------------------
    def match_book(self, order: Order, timestamp: int = 0, is_mine: bool = True) -> List[Trade]:
        """
        Match an incoming taker order against resting orders.

        is_mine=True  (algo taking):  sweep market resting orders up/down to order.price.
                                      Return trades with buyer/seller="SUBMISSION".
        is_mine=False (market taking): sweep market resting first (time priority), then
                                       algo resting at each level. Return only the trades
                                       that fill the algo's resting quantity.

        Consumed resting quantity is removed from the book in both cases.
        """
        trades: List[Trade] = []
        remaining = order.quantity

        if order.quantity > 0:  # taker buy — sweeps asks (ascending)
            all_ask_prices = sorted(set(self.mkt_sell_orders) | set(self.my_sell_orders))
            for ask_price in all_ask_prices:
                if ask_price > order.price or remaining <= 0:
                    break
                if is_mine:
                    # algo buys — hit market asks only
                    avail = abs(self.mkt_sell_orders.get(ask_price, 0))
                    fill = min(remaining, avail)
                    if fill > 0:
                        trades.append(Trade(order.symbol, ask_price, fill,
                                            buyer="SUBMISSION", seller="", timestamp=timestamp))
                        remaining -= fill
                        if fill >= avail:
                            self.mkt_sell_orders.pop(ask_price, None)
                        else:
                            self.mkt_sell_orders[ask_price] += fill
                else:
                    # external buy — market asks fill first, then algo asks
                    mkt_avail = abs(self.mkt_sell_orders.get(ask_price, 0))
                    mkt_fill = min(remaining, mkt_avail)
                    remaining -= mkt_fill
                    if mkt_fill >= mkt_avail:
                        self.mkt_sell_orders.pop(ask_price, None)
                    elif mkt_fill > 0:
                        self.mkt_sell_orders[ask_price] += mkt_fill

                    my_avail = abs(self.my_sell_orders.get(ask_price, 0))
                    my_fill = min(remaining, my_avail)
                    if my_fill > 0:
                        trades.append(Trade(order.symbol, ask_price, my_fill,
                                            buyer="", seller="SUBMISSION", timestamp=timestamp))
                        remaining -= my_fill
                        if my_fill >= my_avail:
                            self.my_sell_orders.pop(ask_price, None)
                        else:
                            self.my_sell_orders[ask_price] += my_fill

        elif order.quantity < 0:  # taker sell — sweeps bids (descending)
            all_bid_prices = sorted(set(self.mkt_buy_orders) | set(self.my_buy_orders), reverse=True)
            for bid_price in all_bid_prices:
                if bid_price < order.price or remaining >= 0:
                    break
                if is_mine:
                    # algo sells — hit market bids only
                    avail = self.mkt_buy_orders.get(bid_price, 0)
                    fill = min(abs(remaining), avail)
                    if fill > 0:
                        trades.append(Trade(order.symbol, bid_price, fill,
                                            buyer="", seller="SUBMISSION", timestamp=timestamp))
                        remaining += fill
                        if fill >= avail:
                            self.mkt_buy_orders.pop(bid_price, None)
                        else:
                            self.mkt_buy_orders[bid_price] -= fill
                else:
                    # external sell — market bids fill first, then algo bids
                    mkt_avail = self.mkt_buy_orders.get(bid_price, 0)
                    mkt_fill = min(abs(remaining), mkt_avail)
                    remaining += mkt_fill
                    if mkt_fill >= mkt_avail:
                        self.mkt_buy_orders.pop(bid_price, None)
                    elif mkt_fill > 0:
                        self.mkt_buy_orders[bid_price] -= mkt_fill

                    my_avail = self.my_buy_orders.get(bid_price, 0)
                    my_fill = min(abs(remaining), my_avail)
                    if my_fill > 0:
                        trades.append(Trade(order.symbol, bid_price, my_fill,
                                            buyer="SUBMISSION", seller="", timestamp=timestamp))
                        remaining += my_fill
                        if my_fill >= my_avail:
                            self.my_buy_orders.pop(bid_price, None)
                        else:
                            self.my_buy_orders[bid_price] -= my_fill

        return trades

    # ------------------------------------------------------------------
    def to_order_depth(self) -> OrderDepth:
        """Return a snapshot of the combined book as an OrderDepth."""
        od = OrderDepth()
        od.buy_orders = self.buy_orders
        od.sell_orders = self.sell_orders
        return od


class Backtester:

    def __init__(self, algorithm, use_full=False):
        self.timestamp = 0

        self.algo = algorithm

        self.trader_data = ""
        # add all new products to listings and position limits.
        self.listings = {
            "INTARIAN_PEPPER_ROOT": Listing(
                symbol="INTARIAN_PEPPER_ROOT",
                product="INTARIAN_PEPPER_ROOT",
                denomination= "XIRECS"
            ),
            "ASH_COATED_OSMIUM": Listing(
                symbol="ASH_COATED_OSMIUM",
                product="ASH_COATED_OSMIUM",
                denomination= "XIRECS"
            ),
            }

        self.position_limits = {
            "INTARIAN_PEPPER_ROOT": 80,
            "ASH_COATED_OSMIUM ": 80,
        }
        self.own_trades = {p: [] for p in self.listings}
        self.position = {p: 0 for p in self.listings}



        # load backtest data (multiple days combined)

        self.iterations = 30000
        if use_full:
            prices_file = 'prices_combined'
            trades_file = 'trades_combined'
        else:
            prices_file = 'prices_short'
            trades_file = 'trades_short'
            self.iterations = 2000

        self.order_data = pd.read_csv(f'data/round1/{prices_file}.csv', header=0, sep=';')

        self.trades_data = pd.read_csv(f'data/round1/{trades_file}.csv', header=0, sep=';')



        # critical data logs
        self.trade_log = {p: [] for p in self.listings}
        self.my_book_log = {p: [] for p in self.listings}

    def step(self):

        # step 1 - build tradingState

        order_book_now = self.order_data[self.order_data['timestamp'] == self.timestamp]

        order_depths = {}

        for product in self.listings:
            order_depth = OrderDepth()
            book_data = order_book_now[order_book_now['product'] == product]
            bid_1 = book_data['bid_price_1'].iloc[0]
            if not pd.isna(bid_1):
                order_depth.buy_orders[int(bid_1)] = int(book_data['bid_volume_1'].iloc[0])

            bid_2 = book_data['bid_price_2'].iloc[0]
            if not pd.isna(bid_2):
                order_depth.buy_orders[int(bid_2)] = int(book_data['bid_volume_2'].iloc[0])

            bid_3 = book_data['bid_price_3'].iloc[0]
            if not pd.isna(bid_3):
                order_depth.buy_orders[int(bid_3)] = int(book_data['bid_volume_3'].iloc[0])

            ask_1 = book_data['ask_price_1'].iloc[0]
            if not pd.isna(ask_1):
                order_depth.sell_orders[int(ask_1)] = -int(book_data['ask_volume_1'].iloc[0])

            ask_2 = book_data['ask_price_2'].iloc[0]
            if not pd.isna(ask_2):
                order_depth.sell_orders[int(ask_2)] = -int(book_data['ask_volume_2'].iloc[0])

            ask_3 = book_data['ask_price_3'].iloc[0]
            if not pd.isna(ask_3):
                order_depth.sell_orders[int(ask_3)] = -int(book_data['ask_volume_3'].iloc[0])

            order_depths[product] = order_depth

        market_trades = {p: [] for p in self.listings}

        plainval_obs = {}
        conv_obs = {}
        observation = Observation(plainval_obs, conv_obs)
        trade_state = TradingState(traderData=self.trader_data,
                                    timestamp=self.timestamp,
                                    listings=self.listings,
                                    order_depths=order_depths,
                                    own_trades=self.own_trades,
                                    market_trades=market_trades,
                                    position=self.position,
                                    observations=observation
                                   )
        # step 2 - run algorithm (get orders)
        user_orders, conversions, trader_data = self.algo.run(trade_state)
        self.trader_data = trader_data

        # flush own trades
        self.own_trades = {p: [] for p in self.listings}

        # track net position change within this step so limits are enforced
        # across both taker fills (step 3) and maker fills (step 4)
        step_delta = {p: 0 for p in self.listings}

        # step 3 - match orders on order book

        # populate order books with market resting orders from the price CSV
        order_books = {p: OrderBook(p) for p in self.listings}
        for product in order_books:
            order_books[product].mkt_buy_orders = dict(order_depths[product].buy_orders)
            order_books[product].mkt_sell_orders = dict(order_depths[product].sell_orders)

        # algo taker matching: cancel any order that would breach position limit,
        # then rest unfilled quantity as a maker order
        for product in user_orders:
            limit = self.position_limits[product]
            order_book = order_books[product]
            for order in user_orders[product]:
                effective_pos = self.position[product] + step_delta[product]
                # cancel the whole order if it would push position beyond the limit
                if order.quantity > 0 and effective_pos + order.quantity > limit:
                    continue
                if order.quantity < 0 and effective_pos + order.quantity < -limit:
                    continue

                trades = order_book.match_book(order, self.timestamp, is_mine=True)
                self.own_trades[product].extend(trades)
                for t in trades:
                    step_delta[product] += t.quantity if t.buyer == "SUBMISSION" else -t.quantity

                # rest unfilled quantity as a maker order
                filled = sum(t.quantity for t in trades)
                unfilled_qty = order.quantity - filled if order.quantity > 0 else order.quantity + filled
                if unfilled_qty != 0:
                    order_book.add_to_book(Order(product, order.price, unfilled_qty), is_mine=True)

            # step 3.5 - save own resting book!
            self.my_book_log[product].append({
                "timestamp": self.timestamp,
                "market_buy_orders": dict(order_depths[product].buy_orders),
                "market_sell_orders": dict(order_depths[product].sell_orders),
                "my_buy_orders": dict(order_book.my_buy_orders),
                "my_sell_orders": dict(order_book.my_sell_orders),
            })

        # step 4 - check if any resting orders are filled against trades.csv
        trades_today = self.trades_data[self.trades_data['timestamp'] == self.timestamp]
        if trades_today is not None:
            for _, trade in trades_today.iterrows():
                symbol = trade['symbol']
                order_book = order_books[symbol]
                bids = order_book.buy_orders
                asks = order_book.sell_orders
                if not bids or not asks:
                    continue
                mid_price = (max(bids) + min(asks)) / 2
                trade_price = int(trade['price'])
                trade_qty = int(trade['quantity'])
                # above mid → incoming buy hit the ask; below mid → incoming sell hit the bid
                if trade_price >= mid_price:
                    incoming = Order(symbol, trade_price, trade_qty)
                else:
                    incoming = Order(symbol, trade_price, -trade_qty)

                algo_fills = order_book.match_book(incoming, self.timestamp, is_mine=False)
                self.own_trades[symbol].extend(algo_fills)
                for t in algo_fills:
                    step_delta[symbol] += t.quantity if t.buyer == "SUBMISSION" else -t.quantity

        # step 5 - update position and record trade history
        for product in self.listings:
            self.position[product] += step_delta[product]

            if len(self.own_trades[product]) != 0:
                self.trade_log[product].append(
                    {
                        "timestamp": self.timestamp,
                        "trades": self.own_trades[product]
                    })

        # any final things
        self.timestamp += 100


    def run(self):
        for i in range(self.iterations):
            self.step()






