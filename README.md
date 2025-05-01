# Air Passengers Time Series Forecasting Using LSTM

##  Project Overview
This project demonstrates how to forecast monthly international airline passenger numbers using a Long Short-Term Memory (LSTM) neural network. The dataset used contains historical data from 1949 to 1960. This notebook performs data visualization, preprocessing, sequence modeling, and future value prediction using an LSTM model built with Keras.

## Dataset
- **File**: `AirPassengers.csv`
- **Columns**:
  - `Month`: Time in YYYY-MM format.
  - `Passengers`: Number of passengers per month.

##  Requirements
- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `keras`
  - `tensorflow`

Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn keras tensorflow
```

##  Steps Performed

1. **Data Loading & Visualization**  
   - Loads CSV data and plots passenger trends over time.

2. **Normalization**  
   - Scales the data using MinMaxScaler for better LSTM performance.

3. **Sequence Preparation**  
   - Converts the time series into a supervised learning problem by creating sequences of 12 months to predict the next month.

4. **Train-Test Split**  
   - 80% of the data is used for training and 20% for testing.

5. **Model Architecture**  
   - A Sequential LSTM model with:
     - 1 LSTM layer (100 units, ReLU activation)
     - 1 Dense output layer
   - Early stopping is used to prevent overfitting.

6. **Training & Evaluation**  
   - Model is trained and validated.
   - Predictions are inverse scaled and compared visually to actual values.

##  Output
- Time series plot showing actual vs. predicted values for the test set.

##  Notes & Improvements
- Try other LSTM variants like **Bidirectional LSTM** or **Stacked LSTM** for performance gains.
- Consider **multi-step forecasting** or **seasonal decomposition** to improve accuracy.
- Hyperparameter tuning (e.g., units, dropout, sequence length) can further optimize performance.

##  Theoretical Addendum

### RNN/LSTM Configurations:

| Configuration | Description | Use Case |
|---------------|-------------|----------|
| **One-to-One** | Single input to single output (e.g., image classification) | Simple classification |
| **One-to-Many** | One input to a sequence of outputs | Image captioning |
| **Many-to-One** | Sequence of inputs to a single output | Time series forecasting (e.g., this project) |
| **Many-to-Many (synchronized)** | Input and output sequences are of the same length | Named Entity Recognition |
| **Many-to-Many (unsynchronized)** | Input and output sequences of different lengths | Machine Translation |

### Bidirectional LSTM:
- Processes sequences in both forward and backward directions.
- Useful when the entire sequence is available.
- **Pros**: Richer context
- **Cons**: Higher computation, not suited for real-time prediction

### Stacked LSTM:
- Multiple LSTM layers stacked to increase depth.
- **Pros**: Learns hierarchical patterns
- **Cons**: Increased training time, overfitting risk, more sensitive to vanishing gradients
