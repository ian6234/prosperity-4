mid = top_mid('ASH_COATED_OSMIUM', state)
if mid != -1:
    if len(trader_data['ASH_COATED_OSMIUM']['mid_price']) <= 29:
        trader_data['ASH_COATED_OSMIUM']['mid_price'].append(mid)
    elif len(trader_data['ASH_COATED_OSMIUM']['mid_price']) == 30:
        if trader_data['ASH_COATED_OSMIUM']['last_ema'] == -1:
            sma = simple_moving_average(trader_data['ASH_COATED_OSMIUM']['mid_price'])
            trader_data['ASH_COATED_OSMIUM']['last_ema'] = exp_moving_average(mid, sma, 30)
        else:
            last_ema = trader_data['ASH_COATED_OSMIUM']['last_ema']
            trader_data['ASH_COATED_OSMIUM']['last_ema'] = exp_moving_average(mid, last_ema, 30)




class VelvetOptions(MultiProductStrategy):

    STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
    # fit on close to money options
    STRIKES_FIT = [5000, 5100, 5200, 5300, 5400, 5500]
    ROUND_TTE = 8  # 8 for testing, 5 for round 3, 4 for round 4, ...
    TICKS_PER_ROUND = 10000
    MIN_IV = 0.001
    TAKER_THRESHOLD = 2

    # parameters for hedging aggression
    HEDGE_THRESHOLD = 25
    URGENT_THRESHOLD = 75

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

        for _ in range(iterations):
            diff = self.call_price(S, strike, time, r, sigma_est) - opt_price
            try:
                vega = self.call_vega(S, strike, time, r, sigma_est)
                if vega < 1e-10:  # vega near zero = unstable, abort
                    return -1
                sigma_est -= diff / vega
                sigma_est = max(0.001, min(sigma_est, 3.0))  # clamp to sane range
            except:
                return -1
            if abs(diff) < 0.001:
                break
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
            buy_thresh = -3 if not need_to_sell else 999
            sell_thresh = -3 if need_to_sell else 999

        orders.extend(
            self.taker_execution(
                'VELVETFRUIT_EXTRACT', S,
                buy_thresh, sell_thresh,
                order_budgets, implied_positions, resting_book, max_quantity=int(abs(net_delta))
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

        # DEBUG - log surface fit and deviations for all ticks

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
