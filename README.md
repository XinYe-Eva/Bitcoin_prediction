# 670FinalProject

\author{
  Chongdan Pan\\
  \texttt{pandapcd@umich.edu}
  \and
  Xin Ye\\
  \texttt{xinye@umich.edu}
\and
  Fangzhe Li\\
  \texttt{fangzhel@umich.edu}
}

\begin{abstract}
Bitcoin and cryptocurrency have been hot topics these years due to the rapid growth of the market value. Due to the lack of regulation, the market of cryptocurrency is very volatile and risky for investors. This project aims to construct a systematic way to predict the volatility of the Bitcoin market through the microstructure of the market. The data source is a high-frequency orderbook, which is a snapshot of all the orders listed on the exchange, and they're collected from Tardis, a major cryptocurrency data collector. Various machine learning models and features will be investigated and their performance will be evaluated by different metrics as well.
\end{abstract}

\section{Introduction}
\par Cryptocurrency such as Bitcoin and Ethereum is a novel digital currencies that have increasingly become a popular topic in the field of economy. They are “digital assets designed to work as a medium of exchange using cryptography to secure the transactions and to control the creation of additional units of the currency” [4]. More people have started investing in cryptocurrencies while they are also notable for being less stable and with stronger volatility [5]. There are two reasons for the rapid growth of the cryptocurrency market. First, since cryptocurrency is running in a decentralized, there is no restrictive regulation stopping investors from entering and exiting the market. Such freedom greatly attracts retail investors as well as financial institutions. Second, cryptocurrency is extremely volatile and opens 24/7 every year. The volatility and long trading hour imply that there are countless opportunities for traders to make money. However, with high returns, there must be high risk, which should never be ignored by investors. The new topic of forecasting cryptocurrencies’ volatility has increasingly been an important topic in the finance area but it is not well-studied and fully explored. Therefore, this project aims to contribute to the prediction of the volatility of Bitcoin to not only help make a profit in the cryptocurrency market but also try to dig out more meaningful information from the market.


\par In this project, we're interested in studying the microstructure of the market of cryptocurrencies. Orderbook is the main source of the features we used, which is a snapshot of any market that stores all "buy" or "sell" orders information such as price and volume. Exchanges are usually made by matching orders from the price of the orderbook between buyer and seller. The orderbook data can provide insights into detailed trading information. Orderbook is generated extremely frequently because as long as anyone posts an order at the market, a new row of orderbook will be made. Therefore, the orderbook can be the detailed information source of the price movement. In this project, we used what we learned in SI 670, explored the bid price and ask price provided in orderbook, and constructed new predictors with feature engineering. By applying linear regression with ridge regularization and KernelPCA together, we successfully beat our baseline, which is Generalized AutoRegressive Conditional Heteroskedasticity(GARCH), a statistical model typically used to predicate the volatility based on the historical return. In addition, we explore other different models, including normal linear regression, XGBoost tree, and different time series models like LSTM and GRU. Finally, it turns out that our model can provide valuable insights through dynamic features from orderbook to explain Bitcoin price fluctuations.

