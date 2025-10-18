import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

class STLDecomposition:
    def __init__(self, data=None, period=12):
        self.data = data
        self.period = period
        
    def load_data(self, file_path):  
        self.data = pd.read_csv(file_path, sep="\t")
        return self.data
    
    def preprocess(self):
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        split_col = self.data["freq,unit,coicop,geo\\TIME_PERIOD"].str.split(",", expand=True)
        split_col.columns = ["freq", "unit", "coicop", "geo"]
        data = pd.concat([split_col, self.data.drop(columns=["freq,unit,coicop,geo\\TIME_PERIOD"])], axis=1)
        data = data[data["geo"] == "DE"]
        data = data[data["coicop"] == "CP01"]
        data = data.T
        data = data.iloc[4:, :]
        data = data.rename(columns={data.columns[0]: "HICP"}).reset_index()
        data = data.rename(columns={"index": "date"})
        return data
    
    def add_feature(self, data):
        data["HICP"] = pd.to_numeric(data["HICP"], errors="coerce")
        data["growth_rate"] = data.HICP.pct_change(fill_method=None) * 100
        data = data.dropna()
        data["date"] = pd.to_datetime(data["date"], format='mixed')
        return data
        
    def decompose(self, data, series_col="HICP"):
        stl = STL(data[series_col], period=self.period)
        result = stl.fit()
        return result


    def plot(self, data, result):       
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        ax1.plot(data["date"], data.HICP)
        ax1.set_title('Original Sales Data')
        ax2.plot(data["date"], result.trend)
        ax2.set_title('Trend Component')
        ax3.plot(data["date"], result.seasonal)
        ax3.set_title('Seasonal Component')
        ax4.plot(data["date"], result.resid)
        ax4.set_title('Residual (Noise) Component')
        plt.tight_layout()
        plt.show()