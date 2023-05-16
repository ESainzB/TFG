import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,r2_score
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
import yfinance as yf

class LstmModel:
    """
    Class that implements a BiLSTM model to predict the price of an asset and make transactions based
    on the predictions
    """
    def __init__(self, ticker, timeframe='1d', cash=100000):

        self.ticker = ticker
        self.timeframe = timeframe
        self.cash = cash
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.rsi_value = []
        self.model_predictions = np.array([])
        self.position = 0
        self.shares = 0
        self.open_price = 0
        self.model = None
        self.window_size = 100
        self.start = dt.datetime(2013,1,1)
        self.end = dt.datetime(2023,1,1)

    def get_data(self):
        """
        Download data into a DataFrame.
        """
        self.data = yf.download(self.ticker, start=self.start, end=self.end, interval=self.timeframe)

    def transform_data(self):
        """
        Drop columns, create Moving Averages and split data.
        """
        self.data = self.data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.data['SmallMA'] = self.data['Adj Close'].rolling(window=5).mean()
        self.data['BigMA'] = self.data['Adj Close'].rolling(window=10).mean()
        self.train, self.test = train_test_split(self.data, test_size=0.2, shuffle=False)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_scaled = self.scaler.fit_transform(self.train['Adj Close'].values.reshape(-1, 1))
        self.test_scaled = self.scaler.fit_transform(self.test['Adj Close'].values.reshape(-1, 1))
        self.portfolio_value = pd.Series(index=self.test.index[100:], dtype='float64') 
        self.trades_track = pd.Series(index=self.test.index, dtype='str')
        self.returns = pd.Series(index=self.test.index[100:], dtype='float64')

    def plot_data(self):
        """
        Plots train and test data.
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

    def prepare_data(self):
        """
        Prepare data
        """
        for i in range(self.window_size, len(self.train_scaled)):
            self.X_train.append(self.train_scaled[i - self.window_size : i, 0])
            self.y_train.append(self.train_scaled[i, 0])
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)

        for i in range(self.window_size, len(self.test_scaled)):
            self.X_test.append(self.test_scaled[i - self.window_size : i, 0])
            self.y_test.append(self.test_scaled[i, 0])
        self.X_test, self.y_test = np.array(self.X_test), np.array(self.y_test)

        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))
        self.y_train = np.reshape(self.y_train, (self.y_train.shape[0], 1))
        self.y_test = np.reshape(self.y_test, (self.y_test.shape[0], 1))

    def create_model(self, lstm_units=50, learning_rate=0.001):
        model = Sequential()
        model.add(Bidirectional(LSTM(lstm_units), input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def optimize_params(self):
        param_dist = {'lstm_units': [50, 75, 100],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'epochs': [100, 150, 200],
                    'batch_size': [32, 64, 128]}
        model = KerasRegressor(build_fn=self.create_model, verbose=0)
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, scoring='neg_mean_squared_error', cv=5)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        random_search_result = random_search.fit(self.X_train, self.y_train, epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stop])
        self.best_params = random_search_result.best_params_
        print(self.best_params)

    def train_model(self):
        """
        Trains the biLSTM model using train data and stores the prediction.
        """
        #self.model = self.create_model(lstm_units=self.best_params['lstm_units'], learning_rate=self.best_params['learning_rate'])
        #self.model.fit(self.X_train, self.y_train, epochs=self.best_params['epochs'], batch_size=self.best_params['batch_size'])
        self.model = self.create_model(lstm_units=50, learning_rate=0.01)
        self.model.fit(self.X_train, self.y_train, epochs=150, batch_size=64)
        self.model_predictions = self.model.predict(self.X_test)
        self.model_predictions = self.scaler.inverse_transform(self.model_predictions)
        self.y_test = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        print(self.model.summary())
        mse = mean_squared_error(self.test['Adj Close'][100:], self.model_predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(self.test['Adj Close'][100:], self.model_predictions)
        r2 = r2_score(self.test['Adj Close'][100:], self.model_predictions)
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}")
        print(f"R cuadrado: {r2:.4f}")

    def plot_predictions(self):
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(self.test.index[100:], self.y_test, label='Test')
        ax.plot(self.test.index[100:], self.model_predictions, label='Predicciones')
        ax.plot(self.test.index[100:], self.test.SmallMA[100:], label='Media pequeña', alpha=0.3)
        ax.plot(self.test.index[100:], self.test.BigMA[100:], label='Media grande', alpha=0.3)
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
        close_prices = self.test[100:]['Adj Close']
        self.rsi_value = ta.momentum.RSIIndicator(close_prices, rsi_period).rsi()

        for i in range(len(self.y_test)):
            prediction = self.model_predictions[i]
            true_test_value = self.y_test[i]
            rsi = self.rsi_value[i]

            # Open order
            if prediction > true_test_value and self.position != 1 and self.test.SmallMA[100 + i] > self.test.BigMA[100 + i] and rsi < rsi_oversold:
                shares_to_buy = self.cash // true_test_value
                self.shares += shares_to_buy
                self.cash -= shares_to_buy * true_test_value
                self.position = 1
                self.trades_track[self.test.index[100 + i]] = 'Buy'
                self.open_price = true_test_value

            # Stop Loss order
            elif self.position != -1 and self.shares > 0 and true_test_value < (0.99 * self.open_price):
                self.cash += self.shares * true_test_value
                self.shares = 0
                self.position = -1
                self.trades_track[self.test.index[100 + i]] = 'Sell'
            
            # Take Profit order
            elif self.position != -1 and self.shares > 0 and true_test_value >= (1.02 * self.open_price):
                self.cash += self.shares * true_test_value
                self.shares = 0
                self.position = -1
                self.trades_track[self.test.index[100 + i]] = 'Sell'
                
            else:
                self.trades_track[self.test.index[100 + i]] = 'No trade'

            self.portfolio_value[self.test.index[100 + i]] = self.cash + (self.shares * true_test_value)

        # print(self.trades_track)
         # print(self.portfolio_value)

    def plot_results_data(self):
        """
        Plots the transactions carried out over the test data.
        """
        fig, (ax) =plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
        ax[0].plot(self.test.index[100:], self.test['Adj Close'][100:], label='Test')
        ax[0].plot(self.test.index[100:], self.test.SmallMA[100:], label='Media pequeña', alpha=0.3)
        ax[0].plot(self.test.index[100:], self.test.BigMA[100:], label='Media grande', alpha=0.3)
        ax[0].set_ylabel('Precio ($)')
        ax[0].set_title(f'{self.ticker} Apertura y cierre de operaciones')
        ax[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        for i in range(len(self.test[100:])):
            date = self.test.index[100 + i]
            if self.trades_track[date] == 'Buy':
                ax[0].axvline(x=date, color='g', linestyle='--', label='Buy', alpha=0.3)
            elif self.trades_track[date] == 'Sell':
                ax[0].axvline(x=date, color='r', linestyle='--', label='Sell', alpha=0.3)

        ax[1].plot(self.test.index[100:], self.rsi_value, label='RSI')
        ax[1].set_xlabel('Fecha')
        ax[1].set_ylabel('RSI')
        ax[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        plt.show()

    def plot_results_portfolio(self):
        """
        Plots the evolution of the portfolio value over time and the transactions carried out.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(self.test.index[100:], self.portfolio_value)
        ax.set_ylabel('Valor del portfolio ($)')
        ax.set_title(f'{self.ticker} Valor del portfolio en el tiempo')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        for i in range(len(self.test[100:])):
            date = self.test.index[100 + i]
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
    # Create instances of LstmModel for each asset
    #gold = LstmModel(ticker='GC=F', timeframe='1d')
    #nasdaq = LstmModel(ticker='^IXIC', timeframe='1d')
    eurusd = LstmModel(ticker='EURUSD=X', timeframe='1d')

    # Download data
    #gold.get_data()
    #nasdaq.get_data()
    eurusd.get_data()

    # Prepare data
    #gold.transform_data()
    #gold.prepare_data()
    #nasdaq.transform_data()
    #nasdaq.prepare_data()
    eurusd.transform_data()
    eurusd.prepare_data()

    # Plot data
    #gold.plot_data()
    #nasdaq.plot_data()
    #eurusd.plot_data()

    # Train each model
    #gold.optimize_params()
    #gold.train_model()
    #gold.plot_predictions()
    #nasdaq.optimize_params()
    #nasdaq.train_model()
    #nasdaq.plot_predictions()
    #eurusd.optimize_params()
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
