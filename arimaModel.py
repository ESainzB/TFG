import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import ta
from pmdarima.arima import auto_arima
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,r2_score
import matplotlib.pyplot as plt

class ArimaModel:
    """
    Class that implements an ARIMA model to predict the price of an asset and make transactions based
    on the predictions
    """
    def __init__(self, ticker, timeframe='1d', cash=100000):

        self.ticker = ticker
        self.timeframe = timeframe
        self.cash = cash
        self.model_predictions = []
        self.position = 0
        self.shares = 0
        self.open_price = 0
        self.model = None
        self.start = dt.datetime(2013,1,1)
        self.end = dt.datetime(2023,1,1)

    def get_data(self):
        """
        Download data into a DataFrame.
        """
        self.data = yf.download(self.ticker, start=self.start, end=self.end, interval=self.timeframe)

    def prepare_data(self):
        """
        Drop columns, create Moving Averages and split data.
        """
        self.data = self.data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.data['SmallMA'] = self.data['Adj Close'].rolling(window=5).mean()
        self.data['BigMA'] = self.data['Adj Close'].rolling(window=10).mean()
        self.train, self.test = train_test_split(self.data, test_size=0.2, shuffle=False)
        self.history = [x for x in self.train['Adj Close']]
        self.portfolio_value = pd.Series(index=self.test.index, dtype='float64') 
        self.trades_track = pd.Series(index=self.test.index, dtype='str')
        self.returns = pd.Series(index=self.test.index, dtype='float64')


    def plot_data(self):
        """
        Plots train and test data along with moving averages.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(self.train.index, self.train['Adj Close'], label='Train')
        ax.plot(self.test.index, self.test['Adj Close'], label='Test')
        ax.plot(self.data.index, self.data['SmallMA'], label='Media pequeña', alpha=0.3)
        ax.plot(self.data.index, self.data['BigMA'], label='Media grande', alpha=0.3)
        ax.set_ylabel('Precio ($)')
        ax.set_title(f'{self.ticker} División de datos')
        ax.legend()
        plt.show()

    def train_model(self):
        """
        Trains the ARIMA model using train data and stores the prediction.
        """
        self.model = auto_arima(self.history, seasonal=False, suppress_warnings=True, stepwise=True)
        for i in range(len(self.test)):
            self.model = self.model.fit(self.history)
            yhat = self.model.predict(n_periods=1)[0]
            self.model_predictions.append(yhat)
            true_test_value = self.test.iloc[i]['Adj Close']
            self.history.append(true_test_value)
            #print('predicted=%f, expected=%f' % (yhat, true_test_value))
        
        print(self.model.summary())
        mse = mean_squared_error(self.test['Adj Close'], self.model_predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(self.test['Adj Close'], self.model_predictions)
        r2 = r2_score(self.test['Adj Close'], self.model_predictions)
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}")
        print(f"R cuadrado: {r2:.4f}")
    
    def plot_predictions(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(self.test.index, self.test['Adj Close'], label='Test')
        ax.plot(self.test.index, self.model_predictions, label='Predicciones')
        ax.plot(self.test.index, self.test.SmallMA, label='Media pequeña', alpha=0.3)
        ax.plot(self.test.index, self.test.BigMA, label='Media grande', alpha=0.3)
        ax.set_ylabel('Precio ($)')
        ax.set_title(f'{self.ticker} Datos reales y Predicciones')
        ax.legend()
        plt.show()
            
    def execute_trades(self):
        """
        Executes buying and selling transactions based on model predictions.
        """
        rsi_period = 14
        rsi_overbought = 70
        rsi_oversold = 50
        close_prices = self.test['Adj Close']
        self.test['RSI'] = ta.momentum.RSIIndicator(close_prices, rsi_period).rsi()

        for i in range(len(self.test)):
            prediction = self.model_predictions[i]
            true_test_value = self.test['Adj Close'][i]
            rsi = self.test['RSI'][i]

            # Open order
            if prediction > true_test_value and self.position != 1 and self.test.SmallMA[i] > self.test.BigMA[i] and rsi < rsi_oversold:
                shares_to_buy = self.cash // true_test_value
                self.shares += shares_to_buy
                self.cash -= shares_to_buy * true_test_value
                self.position = 1
                self.trades_track[self.test.index[i]] = 'Buy'
                self.open_price = true_test_value

            # Stop Loss order
            elif self.position != -1 and self.shares > 0 and true_test_value < (0.99 * self.open_price):
                self.cash += self.shares * true_test_value
                self.shares = 0
                self.position = -1
                self.trades_track[self.test.index[i]] = 'Sell'
            
            # Take Profit order
            elif self.position != -1 and self.shares > 0 and true_test_value >= (1.02 * self.open_price):
                self.cash += self.shares * true_test_value
                self.shares = 0
                self.position = -1
                self.trades_track[self.test.index[i]] = 'Sell'
                
            else:
                self.trades_track[self.test.index[i]] = 'No trade'

            self.portfolio_value[self.test.index[i]] = self.cash + (self.shares * true_test_value)

        # print(self.trades_track)
         # print(self.portfolio_value)

    def plot_results_data(self):
        """
        Plots the transactions carried out over the test data.
        """
        fig, (ax) =plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
        ax[0].plot(self.test.index, self.test['Adj Close'], label='Test')
        ax[0].plot(self.test.index, self.test.SmallMA, label='Media pequeña', alpha=0.3)
        ax[0].plot(self.test.index, self.test.BigMA, label='Media grande', alpha=0.3)
        ax[0].set_ylabel('Precio ($)')
        ax[0].set_title(f'{self.ticker} Apertura y cierre de operaciones')
        ax[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        for i in range(len(self.test)):
            date = self.test.index[i]
            if self.trades_track[date] == 'Buy':
                ax[0].axvline(x=date, color='g', linestyle='--', label='Buy', alpha=0.3)
            elif self.trades_track[date] == 'Sell':
                ax[0].axvline(x=date, color='r', linestyle='--', label='Sell', alpha=0.3)

        ax[1].plot(self.test.index, self.test.RSI, label='RSI')
        ax[1].set_xlabel('Fecha')
        ax[1].set_ylabel('RSI')
        ax[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        plt.show()
            
    def plot_results_portfolio(self):
        """
        Plots the evolution of the portfolio value over time and the transactions carried out.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(self.test.index, self.portfolio_value)
        ax.set_ylabel('Valor del portfolio ($)')
        ax.set_title(f'{self.ticker} Valor del portfolio en el tiempo')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        for i in range(len(self.test)):
            date = self.test.index[i]
            if self.trades_track[date] == 'Buy':
                ax.axvline(x=date, color='g', linestyle='--', label='Buy', alpha=0.3)
            elif self.trades_track[date] == 'Sell':
                ax.axvline(x=date, color='r', linestyle='--', label='Sell', alpha=0.3)
        plt.show()

    def compute_returns(self):
        """
        Computes the daily returns of the portfolio based on the transactions executed.
        """
        self.returns[0] = 0.0
        for i in range(1, len(self.returns)):
            prev_value = self.portfolio_value.iloc[i - 1]
            curr_value = self.portfolio_value.iloc[i]
            self.returns.iloc[i] = (curr_value - prev_value) / prev_value
    
    def compute_drawdown(self):
        """
        Computes the drawdown of the portfolio.
        """
        cum_returns = (1 + self.returns).cumprod()
        max_returns = cum_returns.cummax()
        drawdown = (cum_returns - max_returns) / max_returns
        return drawdown
    
    def compute_metrics(self):
        """
        Computes various performance metrics of the portfolio.
        """
        self.compute_returns()
        drawdown = self.compute_drawdown()
        max_drawdown = drawdown.min()
        total_return = (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) - 1.0
        annualized_return = ((1 + total_return) ** (252 / len(self.portfolio_value))) - 1.0
        sharpe_ratio = np.sqrt(252) * self.returns.mean() / self.returns.std()
        
        print(f"Maximo drawdown: {max_drawdown:.2%}")
        print(f"Retorno total: {total_return:.2%}")
        print(f"Retorno anualizado: {annualized_return:.2%}")
        print(f"Sharpe ratio: {sharpe_ratio:.2f}")
    
def main():
    # Create instances of ArimaModel for each asset
    #gold = ArimaModel(ticker='GC=F', timeframe='1d')
    #nasdaq = ArimaModel(ticker='^IXIC', timeframe='1d')
    eurusd = ArimaModel(ticker='EURUSD=X', timeframe='1d')

    # Download data
    #gold.get_data()
    #nasdaq.get_data()
    eurusd.get_data()

    # Prepare data
    #gold.prepare_data()
    #nasdaq.prepare_data()
    eurusd.prepare_data()

    # Plot data
    #gold.plot_data()
    #nasdaq.plot_data()
    eurusd.plot_data()

    # Train each model
    #gold.train_model()
    #gold.plot_predictions()
    #nasdaq.train_model()
    #nasdaq.plot_predictions()
    eurusd.train_model()
    eurusd.plot_predictions()

    # Execute trades for each asset
    #gold.execute_trades()
    #nasdaq.execute_trades()
    eurusd.execute_trades()

    # Evaluate performance of each model
    #gold.plot_results_data()
    #gold.plot_results_portfolio()
    #gold.compute_metrics()
    #nasdaq.plot_results_data()
    #nasdaq.plot_results_portfolio()
    #nasdaq.compute_metrics()
    eurusd.plot_results_data()
    eurusd.plot_results_portfolio()
    eurusd.compute_metrics()
    
if __name__ == '__main__':
    main()
