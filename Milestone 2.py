import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    norm, expon, uniform, pareto, lognorm, chi2, cauchy, invgauss, levy, burr,
    beta, gamma, weibull_min, weibull_max, triang, genextreme,laplace,
    logistic, f, rayleigh, exponweib, exponnorm, exponpow, gompertz, genlogistic
)
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Train_data.csv')
numerical = df.select_dtypes(include=[np.number])
categorical = df.select_dtypes(exclude=[np.number])
data_set = df.drop(columns=['class'])
split_index = int(0.7 * len(data_set))
train_data = numerical[:split_index]
test_data = numerical[split_index:]
test_anomaly_class = df['class'][split_index:].apply(lambda x: 1 if x == 'anomaly' else 0)

mean = train_data.mean()
std_dev = train_data.std()
z_scores = (numerical - mean) / std_dev 
thresholds = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
anomalies_by_threshold = {}
metrics = {}

for t in thresholds:
    anomalies = (z_scores.abs() > t)
    anomaly_counts = anomalies.sum(axis=1)
    detected_anomalies = numerical[anomaly_counts > 0] 
    anomalies_by_threshold[t] = detected_anomalies
    test_anomalies = anomaly_counts[split_index:] > 0
    predictions = test_anomalies.astype(int)
    TP = ((predictions == 1) & (test_anomaly_class == 1)).sum()
    TN = ((predictions == 0) & (test_anomaly_class == 0)).sum()
    FP = ((predictions == 1) & (test_anomaly_class == 0)).sum()
    FN = ((predictions == 0) & (test_anomaly_class == 1)).sum() 
    accuracy = (TP + TN) / len(test_anomaly_class)
    precision = TP / (TP + FP) 
    recall = TP / (TP + FN) 
    metrics[t] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
    print(f"\nThreshold: {t}")
    print(f"Number of anomalies detected: {len(detected_anomalies)}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}") 
metrics
# the most important metric is the recall metric as it ensures that most anomalies are detected
# so that if data was detected as anomaly and it's not it will not put the network into danger
# but if an anomaly data wasn't detected it will put the network into danger so we can tolerate
# false negatives but cannot tolerate false positives

distributions = {
    "normal": norm,
    "exponential": expon,
    "uniform": uniform,
    "pareto": pareto,
    "lognormal": lognorm,
    "chi-square": chi2,
    "cauchy": cauchy,
    "inverse_gaussian": invgauss,
    "levy": levy,
    "burr": burr,
    "beta": beta,
    "gamma": gamma,
    "weibull_min": weibull_min,
    "weibull_max": weibull_max,
    "triangular": triang,
    "generalized_extreme_value": genextreme,
    "laplace": laplace,
    "logistic": logistic,
    "f_distribution": f,
    "rayleigh": rayleigh,
    "exponentiated_weibull": exponweib,
    "exponentially_modified_normal": exponnorm,
    "exponential_power": exponpow,
    "gompertz": gompertz,
    "generalized_logistic": genlogistic,
}

def calculateMSE(col, pdf, parameters):
    counts, edges = np.histogram(col, bins=30, density=True)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    pdf_vals = pdf(bin_centers, *parameters)
    return np.mean((counts - pdf_vals) ** 2)

best_fits = {} 
for i in numerical.columns:
    bestMSE = 100000
    bestFit = None
    filtered_col = numerical[i][numerical[i] < numerical[i].quantile(0.99)]
    
    if len(numerical[i].unique()) > 30:
        for name, dist in distributions.items():
            parameters = dist.fit(filtered_col)
            mse = calculateMSE(filtered_col, dist.pdf, parameters)
            if mse < bestMSE:
                bestMSE = mse
                bestFit = name
        
        best_fits[i] = {'Best Distribution': bestFit, 'MSE': bestMSE}
best_fits_df = pd.DataFrame(best_fits).T
pd.set_option('display.float_format', '{:.20f}'.format)
print("\nBest-fitting distributions for each column:")
print(best_fits_df)

custom_limits = {
    'duration': (0, 3),
 'src_bytes': (0, 697.5),
 'dst_bytes': (0, 1325.625),
 'count': (1, 100),
 'srv_count': (1, 42.0),
 'serror_rate': (0.0, 0.2),
 'srv_serror_rate': (0.0, 0.2),
 'rerror_rate': (0.0, 0.1),
 'srv_rerror_rate': (0.0, 0.1),
 'same_srv_rate': (0.0, 0.2),
 'diff_srv_rate': (0.0, 0.15),
 'srv_diff_host_rate': (0.0, 0.15),
 'dst_host_count': (0, 100),
 'dst_host_srv_count': (0, 100),
 'dst_host_same_srv_rate': (0.0, 0.175),
 'dst_host_diff_srv_rate': (0.0, 0.175),
 'dst_host_same_src_port_rate': (0.0, 0.15),
 'dst_host_srv_diff_host_rate': (0.0, 0.05),
 'dst_host_serror_rate': (0.0, 0.15),
 'dst_host_srv_serror_rate': (0.0, 0.15),
 'dst_host_rerror_rate': (0.0, 0.2),
 'dst_host_srv_rerror_rate': (0.0, 0.03)
    
}

for column, fit_info in best_fits.items():
    best_distribution = fit_info['Best Distribution']
    distribution = distributions[best_distribution]
    data_min, data_max = custom_limits[column]
    data_range_extension = (data_max - data_min) * 0.05
    x = np.linspace(data_min - data_range_extension, data_max + data_range_extension, 1000)
    
    filtered_col = numerical[column][numerical[column] < numerical[column].quantile(0.99)]
    parameters = distribution.fit(filtered_col)
    pdf_fitted = distribution.pdf(x, *parameters)
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_col, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Data Histogram')
    plt.plot(x, pdf_fitted, 'r-')
    plt.title(f'{column} - Best fit (Whole Data): {best_distribution}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(data_min - data_range_extension, data_max + data_range_extension)
    plt.show()
    
    anomaly_data = df[df['class'] == 'anomaly']
    filtered_anomaly_col = anomaly_data[column][anomaly_data[column] < anomaly_data[column].quantile(0.99)]
    parameters_anomaly = distribution.fit(filtered_anomaly_col)
    pdf_fitted_anomaly = distribution.pdf(x, *parameters_anomaly)
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_anomaly_col, bins=30, density=True, alpha=0.6, color='red', edgecolor='black', label='Anomaly Data Histogram')
    plt.plot(x, pdf_fitted_anomaly, 'g-')
    plt.title(f'{column} - Best fit (Anomaly Data): {best_distribution}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(data_min - data_range_extension, data_max + data_range_extension)
    plt.show()
    
    normal_data = df[df['class'] == 'normal']
    filtered_normal_col = normal_data[column][normal_data[column] < normal_data[column].quantile(0.99)]
    parameters_normal = distribution.fit(filtered_normal_col)
    pdf_fitted_normal = distribution.pdf(x, *parameters_normal)
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_normal_col, bins=30, density=True, alpha=0.6, color='green', edgecolor='black', label='Normal Data Histogram')
    plt.plot(x, pdf_fitted_normal, 'b-')
    plt.title(f'{column} - Best fit (Normal Data): {best_distribution}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(data_min - data_range_extension, data_max + data_range_extension)
    plt.show()

for i in df.columns:
    if pd.api.types.is_numeric_dtype(df[i]):
        num_unique_values = df[i].nunique()
        if num_unique_values > 30:
            continue
        else:
            plt.figure(figsize=(12, 6))
            value_counts = df[i].value_counts().sort_index() 
            pmf = value_counts / len(df[i])  
            plt.bar(pmf.index, pmf.values, color='skyblue', edgecolor='black')
            plt.title(f"PMF for {i}")
            plt.xlabel(i)
            plt.ylabel('Probability')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.show()
    else:
        num_unique_values = df[i].nunique()
        if num_unique_values > 30:
            continue
        plt.figure(figsize=(12, 6))
        value_counts = df[i].value_counts()
        pmf = value_counts / len(df[i])
        plt.bar(pmf.index.astype(str), pmf.values, color='salmon', edgecolor='black')
        plt.title(f"PMF for {i}")
        plt.xlabel(i)
        plt.ylabel('Probability')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

anomaly_data = df[df['class'] == 'anomaly']
normal_data = df[df['class'] == 'normal']
for col in df.columns:
    if col == 'class':
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        unique_count = df[col].nunique()
        if unique_count > 30:
            continue
    plt.figure(figsize=(12, 6))
    if pd.api.types.is_numeric_dtype(df[col]):
        if unique_count <= 30: 
            pmf = df[col].value_counts()
            categories = pmf.index
            plt.bar(categories, pmf)
            anomaly_pmf = anomaly_data[col].value_counts(normalize=True).reindex(categories, fill_value=0)
            plt.bar(categories, anomaly_pmf, color='red')
    else:
        pmf = df[col].value_counts(normalize=True)
        categories = pmf.index
        plt.bar(categories, pmf)
        anomaly_pmf = anomaly_data[col].value_counts(normalize=True).reindex(categories)
        plt.bar(categories, anomaly_pmf, color='red')
    plt.title(f"PMF for {col} - Conditioned on Anomaly Data")
    plt.xlabel(col)
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 6))
    if pd.api.types.is_numeric_dtype(df[col]):
        if unique_count <= 30:
            pmf = df[col].value_counts()
            categories = pmf.index
            plt.bar(categories, pmf)
            normal_pmf = normal_data[col].value_counts(normalize=True).reindex(categories, fill_value=0)
            plt.bar(categories, normal_pmf, color='green')
    else:
        pmf = df[col].value_counts(normalize=True)
        categories = pmf.index
        plt.bar(categories, pmf)
        normal_pmf = normal_data[col].value_counts(normalize=True).reindex(categories)
        plt.bar(categories, normal_pmf, color='green')
    plt.xlabel(col)
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()









    
    


