import pandas as pd
import numpy as np
import ta 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def technicalIndicatorRSI(data, window):
    
    data["DATETIME"] = pd.to_datetime(data["DATETIME"])
    data.set_index(["DATETIME"], inplace = True)
    data.index.name = "timestamp"
    data["RSI"] = ta.momentum.rsi(data["CLOSE"], window = window)

    return data

def trainTest(data, date1, date2):
    
    train = data.loc[data.index <= date1, :]
    test = data.loc[data.index >= date2, :]
    
    return train, test

def backtest(data, tp, sl, volume, capital):
    
    data.loc[:, "Flag"] = False
    data.loc[:, "Profit"] = 0
    data.loc[data.index[0], "Profit"] = capital

    for i in range(len(data)):

        # Posición corta
        if data.loc[data.index[i], "RSI"] > 70 and not(data.loc[data.index[i], "Flag"]):

            openPrice = data.loc[data.index[i], "CLOSE"]
            priceSL = data.loc[data.index[i], "CLOSE"] + sl / 10000
            priceTP = data.loc[data.index[i], "CLOSE"] - tp / 10000

            j = i + 1

            while True:
                if j <= len(data) - 1:
                    data.loc[data.index[j], "Flag"] = True
                    # Beneficio
                    if data.loc[data.index[j], "CLOSE"] <= priceTP or data.loc[data.index[j], "CLOSE"] >= priceSL:
                        profit = (openPrice - data.loc[data.index[j], "CLOSE"]) * 10000 * volume * 10

                        break

                    j = j + 1

                else:
                    profit = (openPrice - data.loc[data.index[-1], "CLOSE"]) * 10000 * volume * 10
                    break

            if j <= len(data) - 1:
                data.loc[data.index[j], "Profit"] = profit
            else:
                data.loc[data.index[-1], "Profit"] = profit

        # Posición larga
        elif data.loc[data.index[i], "RSI"] < 30 and not(data.loc[data.index[i], "Flag"]):

            openPrice = data.loc[data.index[i], "CLOSE"]
            priceSL = data.loc[data.index[i], "CLOSE"] - sl / 10000
            priceTP = data.loc[data.index[i], "CLOSE"] + tp / 10000

            j = i + 1

            while True: 
                if j <= len(data) - 1:
                    data.loc[data.index[j], "Flag"] = True
                    # Beneficio
                    if data.loc[data.index[j], "CLOSE"] >= priceTP or data.loc[data.index[j], "CLOSE"] <= priceSL:
                        profit = (-openPrice + data.loc[data.index[j], "CLOSE"]) * 10000 * volume * 10

                        break

                    j = j + 1

                else:
                    profit = (-openPrice + data.loc[data.index[-1], "CLOSE"]) * 10000 * volume * 10
                    break

            if j <= len(data) - 1:
                data.loc[data.index[j], "Profit"] = profit
            else:
                data.loc[data.index[-1], "Profit"] = profit
    
    data["Cummulative Profit"] = data["Profit"].cumsum()
    if len(data[data["Cummulative Profit"] < 0]) > 0:
        data = data.loc[: data[data["Cummulative Profit"] < 0].index[0], :][:-1]
        
    return data


def PSO(train):
    
    # Número de partículas
    pop = 20

    # Parámetros del Stop-Loss 
    # Posición inicial, Valor inicial del mejor global, Mejores posiciones locales
    #x1p = np.random.uniform(low = 1, high = 75, size = pop)
    x1p = np.linspace(1, 100, pop) # 1, 125
    x1pg = 0
    x1pl = x1p

    # Parámetros del Take-Profit
    #x2p = np.random.uniform(low = 1, high = 75, size = pop)
    x2p = np.linspace(1, 100, pop) # 1, 125
    x2pg = 0
    x2pl = x2p

    # Parámetros del Volumen
    #x3p = np.random.uniform(low = 0.01, high = 15, size = pop)
    x3p = np.linspace(0.01, 15, pop) # 15
    x3pg = 0
    x3pl = x3p

    # Desempeño
    fxpg = 100000000000000 # Mejor esempeño (global)
    fxpl = np.ones(pop) * fxpg # Mejor desempeño (locales)

    # Velocidad inicial de las partículas
    vx1 = np.zeros(pop)
    vx2 = np.zeros(pop)
    vx3 = np.zeros(pop)

    c1 = 0.05 # Atracción hacia el local actual 0.05
    c2 = 0.05 # Atracción hacia el global acutal 0.05
    a = 10 # Constante de penalización 100000, actual 10

    # Convergencia 50 iters, 10 iters
    iters = 30
    convergence = np.zeros(iters)

    # Enjambre de partículas 
    for k in range(iters):

        # Función de ajuste 
        fx = np.zeros(pop)
        for i in range(pop):
            dfBacktest = backtest(train, x2p[i], x1p[i], x3p[i], 100000)
            returns = dfBacktest["Cummulative Profit"].pct_change().dropna()
            sharpe = np.mean(returns) / np.std(returns)
            g = dfBacktest["Cummulative Profit"][-1]

            fx[i] = -(g) + a * np.max(1 - x2p[i], 0) + a * np.max(1 - x1p[i], 0) + a * np.max(0.01 - x3p[i], 0) + a * np.max(x3p[i] - 25, 0) + a * np.max(x2p[i] - 100, 0) + a * np.max(x1p[i] - 100, 0) # -150

        # Convergencia
        #convergence[k] = np.mean(fx)
        convergence[k] = min(fx)

        # Evaluación del desempeño de cada partícula
        # Mejor global
        val, idx = min(fx), np.argmin(fx)

        if val < fxpg:
            fxpg = val # Nuevo mínimo global
            x1pg = x1p[idx]
            x2pg = x2p[idx]
            x3pg = x3p[idx]

        # Mejor local
        for j in range(pop):
            if fx[j] < fxpl[j]:
                fxpl[j] = fx[j]
                x1pl[j] = x1p[j]
                x2pl[j] = x2p[j]
                x3pl[j] = x3p[j]

        # Velocidades y posiciones
        vx1 = vx1 + c1 * np.random.uniform() * (x1pl - x1p) + c2 * np.random.uniform() * (x1pg - x1p)
        vx2 = vx2 + c1 * np.random.uniform() * (x2pl - x2p) + c2 * np.random.uniform() * (x2pg - x2p)
        vx3 = vx3 + c1 * np.random.uniform() * (x3pl - x3p) + c2 * np.random.uniform() * (x3pg - x3p)
        x1p = vx1 + x1p
        x2p = vx2 + x2p
        x3p = vx3 + x3p
    
    return x2pg, x1pg, x3pg, convergence


def append_close(dfData,dfTrain, dfTest):
    a = []
    b = []
    d = []
    e = []

    for i in range(len(dfTrain["CLOSE"])):
        a.append("Train")

    for i in range(len(dfTest["CLOSE"])):
        b.append("Test")
        
    c = np.append(a,b)
    
    df = pd.DataFrame()
    df.index = np.append(dfTrain.index,dfTest.index)
    df["Price"] = np.append(dfTrain["CLOSE"],dfTest["CLOSE"])
    df["RSI"] = np.append(dfTrain["RSI"],dfTest["RSI"])
    df["Data"] = c
    
    return df

def f_estadisticas_mad(df,rf):
    df = df.reset_index()
    df = df.groupby(df['timestamp'].dt.date).tail(1)
    df['log_ret'] = np.log(df['Cummulative Profit']) - np.log(df['Cummulative Profit'].shift(1))
    df['return_cum'] = df['log_ret'].cumsum()
    sr = ((df['log_ret'].mean()-rf)/df['log_ret'].std()) # traditional sharpe ratio
    i = np.argmax(np.maximum.accumulate(df['Cummulative Profit']) - df['Cummulative Profit']) # MDD end of the period
    j = np.argmax(df['Cummulative Profit'][:i]) # MDD start of period
    x = np.argmax(np.maximum.accumulate(df['Cummulative Profit']) + df['Cummulative Profit']) # MDU end of the period
    y = np.argmax(df['Cummulative Profit'][:x]) # MDU start of period
    mdd = df['Cummulative Profit'].iloc[i]-df['Cummulative Profit'].iloc[j]
    mdu = df['Cummulative Profit'].iloc[x]-df['Cummulative Profit'].iloc[y]
    data = np.array([["sharpe_original",sr,"Sharpe Ratio"], 
                 ["drawdown_capi",df['timestamp'].iloc[j],"Fecha inicial del DrawDown de Capital"],
                 ["drawdown_capi",df['timestamp'].iloc[i],"Fecha final del DrawDown de Capital"], 
                 ["drawdown_capi",mdd,"Máxima pérdida flotante registrada"], 
                 ["drawup_capi",df['timestamp'].iloc[y],"Fecha inicial del DrawUp de Capital"],
                 ["drawup_capi",df['timestamp'].iloc[x],"Fecha final del DrawUp de Capital"],    
                 ["drawup_capi",mdu,"Máxima ganancia flotante registrada"]])
    mad = pd.DataFrame(data,columns=['metrica','valor','descripcion'])
    
    return mad