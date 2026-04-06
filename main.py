from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from backtest_logic import Backtester
from algo_test import Trader

import json
import io
import csv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/parse-log")
async def parse_log():

    position_limits = {"EMERALDS": 80, "TOMATOES": 80}
    with open("data/logs/13035.log") as f:
        data = json.load(f)

    # Parse activitiesLog CSV
    reader = csv.DictReader(io.StringIO(data['activitiesLog']), delimiter=';')
    activities = list(reader)

    products = sorted(set(r['product'] for r in activities))

    price_history = {p: [] for p in products}
    book_snapshots = {p: [] for p in products}
    mid_by_product_ts = {}

    for row in activities:
        product = row['product']
        ts = int(row['timestamp'])
        mid = float(row['mid_price'])
        best_bid = float(row['bid_price_1']) if row['bid_price_1'] else None
        best_ask = float(row['ask_price_1']) if row['ask_price_1'] else None

        mid_by_product_ts[(product, ts)] = mid

        price_history[product].append({
            "timestamp": ts,
            "mid": mid,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": (best_ask - best_bid) if best_bid and best_ask else None
        })

        def parse_level(price_key, vol_key):
            return (int(row[price_key]), int(row[vol_key])) if row[price_key] and row[vol_key] else None

        market_buy = {}
        market_sell = {}
        for i in ['1', '2', '3']:
            lvl = parse_level(f'bid_price_{i}', f'bid_volume_{i}')
            if lvl:
                market_buy[lvl[0]] = lvl[1]
            lvl = parse_level(f'ask_price_{i}', f'ask_volume_{i}')
            if lvl:
                market_sell[lvl[0]] = lvl[1]

        book_snapshots[product].append({
            "timestamp": ts,
            "market_buy_orders": market_buy,
            "market_sell_orders": market_sell,
            "my_buy_orders": {},
            "my_sell_orders": {}
        })

    # Parse tradeHistory into own_trades and market_trades
    own_trades = {p: [] for p in products}
    market_trades = {p: [] for p in products}

    for trade in data['tradeHistory']:
        symbol = trade['symbol']
        entry = {
            "timestamp": int(trade['timestamp']),
            "price": trade['price'],
            "quantity": trade['quantity'],
        }
        if trade['buyer'] == 'SUBMISSION':
            own_trades[symbol].append({**entry, "side": "BUY"})
        elif trade['seller'] == 'SUBMISSION':
            own_trades[symbol].append({**entry, "side": "SELL"})
        else:
            market_trades[symbol].append({**entry, "side": "BUY"})

    # Build chart_data (one entry per trade-timestamp, tracking running position & pnl)
    chart_data = {p: [] for p in products}
    total_profit = 0

    for product in products:
        position = 0
        cash = 0

        trades_by_ts = {}
        for trade in own_trades[product]:
            trades_by_ts.setdefault(trade['timestamp'], []).append(trade)

        for ts in sorted(trades_by_ts):
            for trade in trades_by_ts[ts]:
                if trade['side'] == 'BUY':
                    position += trade['quantity']
                    cash -= trade['quantity'] * trade['price']
                else:
                    position -= trade['quantity']
                    cash += trade['quantity'] * trade['price']

            mid = mid_by_product_ts.get((product, ts), 0)
            chart_data[product].append({
                "timestamp": ts,
                "profit": cash + position * mid,
                "position": position
            })

        if chart_data[product]:
            total_profit += chart_data[product][-1]["profit"]

    return {"message": {
        "total_profit": total_profit,
        "products": products,
        "position_limits": position_limits,
        "chart_data": chart_data,
        "book_snapshots": book_snapshots,
        "price_history": price_history,
        "market_trades": market_trades,
        "own_trades": own_trades,
    }}



@app.get("/backtest")
async def run_backtest(use_full: bool = Query(default=False)):
    algo = Trader()
    bt = Backtester(algo, use_full)
    bt.run()

    products = [p for p in bt.listings]
    chart_data = {p: [] for p in bt.listings}

    price_history = {p: [] for p in bt.listings}
    market_trades = {p: [] for p in bt.listings}
    own_trades = {p: [] for p in bt.listings}

    book_snapshots = {p: [] for p in bt.listings}

    total_profit = 0

    for product in bt.trade_log:

        # book history data
        book_snapshots[product] = bt.my_book_log[product]

        # price history data
        for row in bt.order_data[bt.order_data['product'] == product].itertuples():
            price_history[product].append({
                "timestamp": row.timestamp,
                "mid": row.mid_price,
                "best_bid": row.bid_price_1,
                "best_ask": row.ask_price_1,
                "spread": row.ask_price_1-row.bid_price_1
            })

        # market trade history data
        for row in bt.trades_data[bt.trades_data['symbol'] == product].itertuples():
            market_trades[product].append({
                "timestamp": row.timestamp,
                "price": row.price,
                "quantity": row.quantity,
                # assume buy for now since not sure
                "side": "BUY"
            })

        # own trades data
        position = 0
        cash = 0
        for trade_batch in bt.trade_log[product]:

            timestamp = trade_batch['timestamp']
            for trade in trade_batch['trades']:
                side = "SELL"
                if trade.buyer == "SUBMISSION":
                    position += trade.quantity
                    cash -= trade.quantity * trade.price
                    side = "BUY"
                else:
                    position -= trade.quantity
                    cash += trade.quantity * trade.price

                # log every trade for the order book separately
                own_trades[product].append({
                    "timestamp": int(trade.timestamp),
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "side": side
                })

            mid_price = bt.order_data[bt.order_data['product'] == product]
            mid_price = mid_price[mid_price['timestamp'] == int(timestamp)]['mid_price']

            # only log profit and position once per batch of trades.
            chart_data[product].append({
                "timestamp": int(timestamp),
                "profit": cash + position * float(mid_price.values[0]),
                "position": position,
            })


        total_profit += chart_data[product][-1]["profit"]


    return {"message":
            {
                "total_profit": total_profit,
                "products": products,
                "position_limits": bt.position_limits,
                "chart_data": chart_data,

                "book_snapshots": book_snapshots,
                "price_history": price_history,
                "market_trades": market_trades,
                "own_trades": own_trades,
            }
        }
