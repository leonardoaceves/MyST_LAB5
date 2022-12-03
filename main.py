import functions
import data
import visualizations
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def dataHandling(filepath, window = 14, date1 = "2020-01-01", date2 = "2020-02-01"):
    
    datan = data.dataReader(filepath)
    datan = functions.technicalIndicatorRSI(datan, window)
    train, test = functions.trainTest(datan, date1, date2)
    
    return datan, train, test

def optimization(train):
    
    tp, sl, vol, convergence = functions.PSO(train)
    info = pd.DataFrame({"Take Profit" : round(tp, 4), "Stop Loss" : round(sl, 4), "Volume" : round(vol, 4)}, index = ["PSO Train"])
    
    return tp, sl, vol, convergence, info
    
def visualOptimization(conv, tp, sl, vol, train, test):
    
    fig1, fig2, fig3, trainBT, testBT = visualizations.visualsOptimization(conv, tp, sl, vol, train, test)
    
    return fig1, fig2, fig3, trainBT, testBT
    
    
def visualCloseRSI(Data,Train,Test):

    df = functions.append_close(Data,Train,Test)
    
    visualizations.plot_CLOSE_RSI(df)    
    
    
def performance(df, rf):

    mad = functions.f_estadisticas_mad(df, rf)
    
    return mad