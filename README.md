# Stock Predict
Exploration via LSTM (Long Short Term Memory) / GRU (Gated Recurrent Unit)

## Common settings
Data sample: Apple stock price in 5 years history
Inputs (size 5): [Volume, Open, High, Low, Close]
Output: Close price on the next day
Test split: Last 50 days
Sequence size: 5 days
Batch size: 1
Loss function: Mean square error
Optimizer: Adam
Learning rate: 0.001
Random seed: 4

## Explored settings
Model: LSTM / GRU
LSTM num layers: 1...5
LSTM hidden size: 1...6
Activation function: None (Direct Output) / Linear / Sigmoid / Linear => Sigmoid
