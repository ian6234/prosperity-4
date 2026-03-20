from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from backtest_logic import Backtester
from algo_test import Trader

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


@app.get("/backtest")
async def run_backtest(tomato_threshold: int = Query(default=2)):
    algo = Trader()
    bt = Backtester(algo)
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
        book_snapshots[product] = bt.my_book_log

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
