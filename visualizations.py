import numpy as np
import functions
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualsOptimization(conv, tp, sl, vol, train, test):

    # Gráfica de convergencia
    fig1 = make_subplots(specs=[[{"secondary_y": False}]])
    fig1.add_trace(go.Scatter(x = np.arange(len(conv)), 
                              y = conv), secondary_y = False,) #mode = "markers"
    
    fig1.update_yaxes(title_text = "Función objetivo", secondary_y = False)
    fig1.update_layout(title = "Gráfica de Convergencia",  xaxis_title = "Iteración")
    
    trainBT = functions.backtest(train, tp, sl, vol, 100000)
    testBT = functions.backtest(test, tp, sl, vol, 100000)
    
    # Evolución de capital en el conjunto de entrenamiento y prueba
    fig2 = make_subplots(specs=[[{"secondary_y": False}]])
    fig2.add_trace(go.Scatter(x = trainBT.index, 
                              y = trainBT["Cummulative Profit"]), secondary_y = False,)
    
    fig2.update_yaxes(title_text = "USD", secondary_y = False)
    fig2.update_layout(title = "Evolución de Capital: Entrenamiento",  xaxis_title = "Fecha")
    
    
    fig3 = make_subplots(specs=[[{"secondary_y": False}]])
    fig3.add_trace(go.Scatter(x = testBT.index, 
                              y = testBT["Cummulative Profit"]), secondary_y = False,)
    
    fig3.update_yaxes(title_text = "USD", secondary_y = False)
    fig3.update_layout(title = "Evolución de Capital: Prueba",  xaxis_title = "Fecha")
    
    return fig1, fig2, fig3, trainBT, testBT



def plot_CLOSE_RSI(df):
    fig1 = go.Figure()

    for contestant, group in df.groupby("Data"):
        fig1.add_trace(go.Scatter(x=group.index, y=group["Price"], name=contestant))

        fig1.update_layout(
            title="EURUSD Close",
            xaxis_title="TimeStamp",
            yaxis_title="Price",
            legend_title="Data Type"
            )
    fig1.show()

    fig2 = go.Figure()

    for contestant, group in df.groupby("Data"):
        fig2.add_trace(go.Scatter(x=group.index, y=group["RSI"], name=contestant))

        fig2.update_layout(
            title="EURUSD RSI",
            xaxis_title="TimeStamp",
            yaxis_title="RSI",
            legend_title="Data Type"
            )
    fig2.show()

