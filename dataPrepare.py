import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
        self.portfolio_value = pd.Series(index=self.test.index, dtype='float64') 
        self.trades_track = pd.Series(index=self.test.index, dtype='str')
        self.returns = pd.Series(index=self.test.index, dtype='float64')

    def plot_data(self):
        """
        Plots train and test data.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(self.train.index, self.train['Adj Close'], label='Train data')
        ax.plot(self.test.index, self.test['Adj Close'], label='Test data')
        ax.plot(self.data.index, self.data['SmallMA'], label='Small MA', alpha=0.3)
        ax.plot(self.data.index, self.data['BigMA'], label='Big MA', alpha=0.3)
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{self.ticker} Price over time')
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
        self.model = self.create_model(lstm_units=self.best_params['lstm_units'], learning_rate=self.best_params['learning_rate'])
        self.model.fit(self.X_train, self.y_train, epochs=self.best_params['epochs'], batch_size=self.best_params['batch_size'])
        #self.model = self.create_model()
        #self.model.fit(self.X_train, self.y_train, epochs=200, batch_size=32)
        self.model_predictions = self.model.predict(self.X_test)
        self.model_predictions = self.scaler.inverse_transform(self.model_predictions)
        self.y_test = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

    def plot_predictions(self):
        plt.plot(self.y_test, label='Real Data')
        plt.plot(self.model_predictions, label='Predicted')
        plt.legend()
        plt.show()
        



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
    eurusd.optimize_params()
    eurusd.train_model()
    eurusd.plot_predictions()

    # Execute trades for each asset
    #gold.execute_trades()
    #nasdaq.execute_trades()
    #eurusd.execute_trades()

    # Evaluate performance of each model
    #gold.plot_results_data()
    #gold.plot_results_portfolio()
    #gold.compute_metrics()
    #nasdaq.plot_results_data()
    #nasdaq.plot_results_portfolio()
    #nasdaq.compute_metrics()
    #eurusd.plot_results_data()
    #eurusd.plot_results_portfolio()
    #eurusd.compute_metrics()
    
if __name__ == '__main__':
    main()
