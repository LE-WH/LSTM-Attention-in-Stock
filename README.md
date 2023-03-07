# LSTM-Attention-in-Stock
An attempt to predict the future in stock market.


- The model consists of two parallel parts: double-layer LSTM and multi-head attention layers. 
- The data inputs for a single stock are long-term history of itself and short-term history of overall stocks.
- The target is to predict whether the stock price will increase or decrease in the following day.


However the best result of accuracy is 54% on validation set, indicating that the model is too simple or the features are too simple.
So I didn't continue on this project, leaving many functions undone (for example testing part).

But this project consists of data crawling and complete model/experiment structure. After learning more financial stuff in the future I may return to this project.




