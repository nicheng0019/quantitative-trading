# 基础配置
start_date = '2010-01-01'
split_date = '2015-01-01'
end_date = '2018-01-01'
instrument = D.instruments(start_date=start_date, end_date=end_date, market='CN_STOCK_A')

# 获取每年年报公告后的第一个交易日历
trading_days = D.trading_days(market='CN', start_date=start_date, end_date=end_date)
trading_days['month'] = trading_days.date.map(lambda x:x.month)
trading_days['year'] = trading_days.date.map(lambda x:x.year)
groupby_td = trading_days.groupby(['year','month']).apply(lambda x:x.head(1))
first_date_after_financial_report = list(groupby_td[groupby_td['month']==5].date) # 5月第一个交易日
first_date_after_financial_report = [i.strftime('%Y-%m-%d') for i in first_date_after_financial_report] # date转换为str 

# 特征列表
financial_features_fields = ['date','fs_roe_0','fs_bps_0','fs_operating_revenue_ttm_0','fs_current_assets_0','fs_non_current_assets_0',
              'fs_roa_0','fs_total_profit_0','fs_free_cash_flow_0','adjust_factor_0','fs_eps_0','pe_ttm_0','close_0',
              'fs_common_equity_0','fs_net_income_0','market_cap_0','fs_eps_yoy_0','beta_szzs_90_0','fs_net_profit_margin_ttm_0',   
             ]

# 按年获取财务特征数据
def get_financial_features(date,instrument=instrument,fields=financial_features_fields):
    assert type(date) == str  
    df = D.features(instrument, date, date, fields)
    return df

# 获取财务特征数据，采取缓存的形式，可以节省运行时间
def get_financial_features_cache():
    print('获取财务特征数据，并缓存！')
    financial_features  = pd.DataFrame()
    for dt in first_date_after_financial_report:
        df = get_financial_features(dt)
        financial_features = financial_features.append(df)   
    return Outputs(financial_features=DataSource.write_df(financial_features))
m1 = M.cached.v2(run=get_financial_features_cache)
financial_features_df = m1.financial_features.read_df()  
 
# 获取日线特征数据
daily_history_features_fields = ['close','amount','pb_lf']  # 标注也在这里获取
def get_daily_history_features(start_date=start_date,end_date=end_date,instrument=instrument,fields=daily_history_features_fields):
    df = D.history_data(instrument,start_date,end_date,fields)
    return df 

# 按股票groupby 计算日线特征
def calcu_daily_history_features(df):
    df['mean_amount'] = pd.rolling_apply(df['amount'], 22, np.nanmean)/df['amount']
    df['month_1_mom'] = df['close']/df['close'].shift(22)
    df['month_12_mom'] = df['close']/df['close'].shift(252)
    df['volatity'] = pd.rolling_apply(df['close'], 90, np.nanstd)/df['close']
    return df 

# 获取日线特征，采取缓存的形式，可以节省运行时间
def get_daily_features_cache():
    print('获取日线特征数据，并缓存！')
    daily_history_features = get_daily_history_features().groupby('instrument').apply(calcu_daily_history_features)
    return Outputs(daily_features=DataSource.write_df(daily_history_features))
m2 = M.cached.v2(run=get_daily_features_cache)
daily_features_df = m2.daily_features.read_df()  
 
# 财务特征和日线特征合并
result=financial_features_df.merge(daily_features_df, on=['date', 'instrument'], how='inner') 

# 抽取衍生特征
# 资产周转率
result['asset_turnover'] = result['fs_operating_revenue_ttm_0']/(result['fs_non_current_assets_0'] + result['fs_current_assets_0']) 
# 总盈利/总资产
result['gross_profit_to_asset'] = result['fs_total_profit_0']/(result['fs_non_current_assets_0'] + result['fs_current_assets_0']) 
# 自营现金流/总资产
result['cash_flow_to_assets'] = result['fs_free_cash_flow_0']/(result['fs_non_current_assets_0'] + result['fs_current_assets_0'])
# 总收入/价格 
result['sales_yield'] = result['fs_operating_revenue_ttm_0']/result['close_0']
# 现金流/股数/股价
result['cash_flow_yield'] = result['fs_free_cash_flow_0']/(result['fs_common_equity_0']/result['close'])/result['close']
# 营业收入 Sales to EV
result['sales_to_ev'] = result['fs_operating_revenue_ttm_0']/result['fs_common_equity_0']
#  EBITDA to EV 
result['ebitda_to_ev'] = result['fs_net_income_0']/result['fs_common_equity_0']

def judge_positive_earnings(df):   
    if df['adjust_factor_0'] > df['adjust_factor_0_forward']:
        return 1
    else:
        return 0
    
# 构建时序衍生特征函数（# 一年前的pe # 一年前的总收入/价格 # 复权因子哑变量）
def construct_derivative_features(tmp):
    tmp['pe_forward'] = tmp['pe_ttm_0'].shift(1) 
    tmp['sales_yield_forward'] = tmp['sales_yield'].shift(1) 
    tmp['adjust_factor_0_forward'] = tmp['adjust_factor_0'].shift(1)
    tmp['positive_earnings'] = tmp.apply(judge_positive_earnings,axis=1)
    # 标注数据构建
    tmp['label'] = tmp['pb_lf'].shift(-1)
    return tmp 

features_df = result.groupby('instrument').apply(construct_derivative_features)

## 去极值和标准化
# 哪些特征需要进行 去极值和标准化处理
need_deal_with_features = ['fs_free_cash_flow_0',  'fs_eps_0',
       'fs_operating_revenue_ttm_0', 'market_cap_0', 'fs_roe_0',
       'fs_current_assets_0', 'fs_roa_0', 'pe_ttm_0',
       'fs_non_current_assets_0', 'fs_eps_yoy_0', 'fs_bps_0', 'close_0',
       'adjust_factor_0', 'fs_common_equity_0', 'fs_net_income_0',
       'fs_total_profit_0', 'amount', 'pb_lf', 'mean_amount',
       'asset_turnover', 'gross_profit_to_asset', 'cash_flow_to_assets',
       'sales_yield', 'cash_flow_yield', 'sales_to_ev', 'ebitda_to_ev',
       'pe_forward', 'sales_yield_forward', 'adjust_factor_0_forward','label']

# 去极值
def remove_extremum(df,features=need_deal_with_features):
    factor_list = features
    for factor in factor_list:
        df[factor][df[factor] >= np.percentile(df[factor], 95)] = np.percentile(df[factor], 95)
        df[factor][df[factor] <= np.percentile(df[factor], 5)] = np.percentile(df[factor], 5)
    return df

# 标准化
def standardization(df,features=need_deal_with_features):
    factor_list = features
    for factor in factor_list:
        df[factor] = (df[factor] - df[factor].mean()) / df[factor].std()
    return df 

def deal_with_features(df):
    return standardization(remove_extremum(df))

features_df_after_deal_with = features_df.groupby('date').apply(deal_with_features)
 
# 整理因子和标注
key_attr = ['instrument', 'date',]
explained_features = [ 'fs_eps_0','fs_net_profit_margin_ttm_0','fs_bps_0','asset_turnover','fs_roa_0','fs_roe_0',
       'gross_profit_to_asset','cash_flow_to_assets', 'positive_earnings', 'pe_forward','sales_yield', 'sales_yield_forward',
        'cash_flow_yield', 'sales_to_ev', 'ebitda_to_ev','market_cap_0',
       'beta_szzs_90_0', 'month_1_mom', 'month_12_mom', 'volatity','mean_amount', 'fs_eps_yoy_0']
label = ['label']
final_data = features_df_after_deal_with[label+key_attr+explained_features +['pb_lf']]

import xgboost as xgb # 导入包

# 样本内的数据同样 划分训练数据和测试数据
assert len(final_data.columns) == 26
data = final_data[final_data['date'] <= '2015-01-01'] # 样本内数据
data = data[~pd.isnull(data[label[0]])] # 删除标注为缺失值的
data = data[key_attr+label+explained_features].dropna()  # 删除 特征为缺失值的
data.index = range(len(data))
train_data = data.ix[:int(len(data)*0.8)]  # 80%的数据拿来训练
test_data = data.ix[int(len(data)*0.8):]   # 剩下的数据拿来验证

# 数据按 特征和标注处理，便于xgboost构建对象
X_train = train_data[explained_features]
y_train = train_data[label[0]]
X_test = test_data[explained_features]
y_test = test_data[label[0]]

# xgboost 构建对象
dtrain = xgb.DMatrix(X_train.values,label=y_train.values) # 这个地方如果是X_train 其实不影响结果
dtest = xgb.DMatrix(X_test.values,label=y_test.values)

# 设置参数，参数的格式用map的形式存储
param = {'max_depth': 1,                  # 树的最大深度
         'eta': 0.1,                        # 一个防止过拟合的参数，默认0.3
         'n_estimators':150,                 # Number of boosted trees to fit 
         'silent': 1,                     # 打印信息的繁简指标，1表示简， 0表示繁
         'objective': 'reg:linear'}  # 使用的模型，分类的数目 

num_round = 100 # 迭代的次数
# 看板，每次迭代都可以在控制台打印出训练集与测试集的损失
watchlist = [(dtest, 'eval'), (dtrain, 'train')]

# 训练模型
bst = xgb.train(param, dtrain, num_round, evals=watchlist)

# 测试集上模型预测
preds = bst.predict(dtest)  
# preds

# 样本外数据 
out_of_sample_data = final_data[final_data['date'] > '2015-01-01'] # 样本内数据
out_of_sample_data = out_of_sample_data[key_attr+explained_features+['pb_lf']]  #  取出 特征数据
assert len(out_of_sample_data.columns) == 25

X_TEST = out_of_sample_data[explained_features]
out_of_sample_dtset = xgb.DMatrix(X_TEST.values) 

# 样本外预测
out_of_sample_preds = bst.predict(out_of_sample_dtset)
out_of_sample_data['predict_pb_lf'] = out_of_sample_preds
out_of_sample_data = out_of_sample_data.dropna()

quit()

# 1. 策略基本参数

# 证券池：这里使用所有股票
instruments = D.instruments()
# 起始日期
start_date = '2015-01-01'
# 结束日期
end_date = '2018-01-01'
# 初始资金
capital_base = 100000
# 策略比较参考标准，以沪深300为例
benchmark = '000300.INDX'
# 调仓周期（多少个交易日调仓）
rebalance_period = 1
# 每轮调仓买入的股票数量
#stock_num = int(len(instruments) * 0.9)



# 2. 选择股票：为了得到更好的性能，在这里做批量计算
# 本样例策略逻辑：选取调仓当天，交易额最小的30只股票买入
# 加载数据：https://bigquant.com/docs/data_history_data.html
history_data = D.history_data(instruments, start_date, end_date, fields=['amount'])
# 过滤掉停牌股票：amount为0的数据
#selected_data = history_data[history_data.amount > 0]
# 按天做聚合(groupby)，对于每一天的数据，做(apply)按交易额升序排列(sort_values)，并选取前30只([:stock_num])
selected_data = out_of_sample_data.groupby('date')


# 3. 策略主体函数
# 初始化虚拟账户状态，只在第一个交易日运行
def initialize(context):
    # 设置手续费，买入时万3，卖出是千分之1.3,不足5元以5元计
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))

def sameday(date1, date2):
    for i in range(3):
        if date1[i] != date2[i]:
            return False
    return True

residual_stocks = []
b_may_first = False
trade_year = -1
# 策略交易逻辑，每个交易日运行一次
def handle_data(context,data):
    global trade_year, b_may_first, residual_stocks
    today = data.current_dt.strftime('%Y-%m-%d') # 交易日期
    #print(today)
    year, month, day = today.split("-")
    #print(year, month, day)
    
    if trade_year != year and month == "05" and not b_may_first:
        trade_year = year
        b_may_first = True
        
    residual_stocks_new = []
    if not b_may_first:
        if len(residual_stocks) != 0:
            for equity in residual_stocks:
                if data.can_trade(equity):
                    context.order_target_percent(equity, 0)
                else:
                    residual_stocks_new.append(equity)
                    
            residual_stocks = residual_stocks_new
        return       
            
    # context.trading_day_index：交易日序号，第一个交易日为0
    #if context.trading_day_index % context.options['rebalance_period'] != 0:
        #return
    intyear = int(year)
    intmonth = int(month)
    intday = int(day)
    print(year, month, day)
    # 调仓：卖出所有持有股票
    for equity in context.portfolio.positions:
        # 停牌的股票，将不能卖出，将在下一个调仓期处理
        if data.can_trade(equity):
            context.order_target_percent(equity, 0)
        else:
            residual_stocks.append(equity)

    # 调仓：买入新的股票
    
    #print(context.options['selected_data'])
    selected_data = context.options['selected_data']
    dates = list(selected_data['date'])
    #print(dates)
    
    startindex = -1
    endindex = -1
    for index, selected_date in enumerate(dates):
        selected_year, selected_month, selected_day = str(selected_date).strip("00:00:00").split("-")
        selected_year = int(selected_year)
        selected_month = int(selected_month)
        selected_day = int(selected_day)
        #print(selected_year, selected_month, selected_day)
        if sameday([selected_year, selected_month, selected_day], [intyear, intmonth, intday]) and startindex == -1:
            startindex = index
        if startindex != -1 and not sameday([selected_year, selected_month, selected_day], [intyear, intmonth, intday]):
            endindex = index
            break
    
    print(startindex, endindex)
    #print(selected_data)
    #data = data.dropna()
    instruments = list(selected_data['instrument'])[startindex:endindex]
    predict_pb_lfs = list(selected_data['predict_pb_lf'])[startindex:endindex]
    pb_lfs = list(selected_data['pb_lf'])[startindex:endindex]
    #print(len(instruments))
    #print(instruments)
    #print(predict_pb_lfs)
    #print(pb_lfs)
    #key = data.keys()
    #print(key)
    
    svalues = []
    instruments_to_buy = []
    market_cap_floats = []
    amounts = []
    for index, instrument in enumerate(instruments):  
        predict_pb_lf = predict_pb_lfs[index]
        pb_lf = pb_lfs[index]
        
        start_date = str(int(year) - 1) + "-" + month + "-" + day
        end_date = today
        df = D.history_data([instrument], start_date, end_date, ['pb_lf', 'market_cap_float'])
        try:
            if len(df) == 0:
                continue
        except:
            #print(df)
            continue
        last_pb_lf = list(df['pb_lf'])
        last_date = list(df['date'])
        market_cap_float = list(df['market_cap_float'])[-1]
        
        start_date = year + "-" + str(intmonth - 1) + "-" + day
        end_date = today
        df = D.history_data([instrument], start_date, end_date, ['amount'])
        one_month_amounts = list(df['amount'])
        
        total_amounts = 0
        for amount in one_month_amounts:
            total_amounts = total_amounts + amount
        
        arpb_lf = np.zeros((12))
        monthindex = -1
        for dateindex, datetemp in enumerate(last_date):
            #print(datetemp)
            lastyear, lastmonth, lastday = str(datetemp).strip(" 00:00:00").split("-")
            if lastmonth != monthindex:
                monthindex = lastmonth
                arpb_lf[int(lastmonth) - 1] = last_pb_lf[dateindex]
        pb_lf_std = np.std(arpb_lf)
        
        svalue = (predict_pb_lf - pb_lf) / pb_lf_std
        
        svalues.append(svalue)
        instruments_to_buy.append(instrument)
        market_cap_floats.append(market_cap_float)
        amounts.append(total_amounts)
        
    #不买市值最小的10%
    market_cap_floats = np.array(market_cap_floats)
    sorted_index = np.argsort(-market_cap_floats)
    num = int(len(sorted_index) * 0.9)
    sorted_index = sorted_index[:num]
     
    svalues = [svalues[i] for i in sorted_index]
    instruments_to_buy = [instruments_to_buy[i] for i in sorted_index]
    amounts = [amounts[i] for i in sorted_index]
    
    #不买过去一个月成交金额最少的10%
    amounts = np.array(amounts)
    sorted_index = np.argsort(-amounts)
    num = int(len(sorted_index) * 0.9)
    sorted_index = sorted_index[:num]
    
    svalues = np.array(svalues)
    sorted_index = np.argsort(-svalues)
    sorted_index = sorted_index[:50]
    print(sorted_index)
    instruments_to_buy = [instruments_to_buy[i] for i in sorted_index]
    print(instruments_to_buy)
    print(svalues[sorted_index])
    
    b_may_first = False
        
    #instruments_to_buy = context.options['selected_data'].ix[today].instrument
    if len(instruments_to_buy) == 0:
        return   
    
    # 等量分配资金买入股票
    weight = 1.0 / len(instruments_to_buy)
    for instrument in instruments_to_buy:
        if data.can_trade(context.symbol(instrument)):
            context.order_target_percent(context.symbol(instrument), weight)

# 4. 策略回测：https://bigquant.com/docs/module_trade.html
m = M.trade.v2(
    instruments=instruments,
    start_date=start_date,
    end_date=end_date,
    initialize=initialize,
    handle_data=handle_data,
    # 买入订单以开盘价成交
    order_price_field_buy='open',
    # 卖出订单以开盘价成交
    order_price_field_sell='open',
    capital_base=capital_base,
    benchmark=benchmark,
    # 传入数据给回测模块，所有回测函数里用到的数据都要从这里传入，并通过 context.options 使用，否则可能会遇到缓存问题
    options={'selected_data': out_of_sample_data, 'rebalance_period': rebalance_period}
)

#m = M.trade.v3(
#    instruments=instruments,
#    options_data=m8.predictions,
#    start_date=start_date,
#    end_date=end_date,
#    handle_data=handle_data,
#    prepare=m12_prepare_bigquant_run,
#    initialize=initialize,
#    volume_limit=0.025,
#    order_price_field_buy='open',
#    order_price_field_sell='open',
#    capital_base=1000000,
#    benchmark='000300.SHA',
#    auto_cancel_non_tradable_orders=True,
#    data_frequency='daily',
#    price_type='后复权',
#    plot_charts=True,
#   backtest_only=False,
#    amount_integer=False
#)
