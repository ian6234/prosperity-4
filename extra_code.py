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