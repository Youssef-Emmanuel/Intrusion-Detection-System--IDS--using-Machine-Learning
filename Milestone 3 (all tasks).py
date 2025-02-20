import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    norm, expon, uniform, pareto, lognorm, chi2, cauchy, invgauss, levy, burr,
    beta, gamma, weibull_min, weibull_max, triang, genextreme, laplace,
    logistic, f, rayleigh, exponweib, exponnorm, exponpow, gompertz, genlogistic
)
from sklearn.naive_bayes import (GaussianNB, MultinomialNB, BernoulliNB)
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Train_data.csv')
numerical = df.select_dtypes(include=[np.number])
categorical = df.select_dtypes(exclude=[np.number])
split_index = int(0.7 * len(df))
train_data = df[:split_index]
test_data = df[split_index:]
test_anomaly_class = test_data['class'].apply(lambda x: 1 if x == 'anomaly' else 0)

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
pmf_fits = {}
for column in numerical.columns:
    bestMSE = 10000
    bestFit = None
    parameters = None 
    filtered_col = train_data[column][train_data[column] < train_data[column].quantile(0.99)]
    if len(filtered_col.unique()) > 30:
        for name, dist in distributions.items():
            params = dist.fit(filtered_col)
            mse = calculateMSE(filtered_col, dist.pdf, params)
            if mse < bestMSE:
                bestMSE = mse
                bestFit = name
                parameters = params
        best_fits[column] = {
            'Best Distribution': bestFit,
            'MSE': bestMSE,
            'Parameters': parameters
        }
    else:
        pmf_fits[column] = {
            "Anomaly": train_data[train_data['class'] == 'anomaly'][column].value_counts(normalize=True).to_dict(),
            "Normal": train_data[train_data['class'] == 'normal'][column].value_counts(normalize=True).to_dict()
        }

for column in categorical.columns:
    pmf_fits[column] = {
        "Anomaly": train_data[train_data['class'] == 'anomaly'][column].value_counts(normalize=True).to_dict(),
        "Normal": train_data[train_data['class'] == 'normal'][column].value_counts(normalize=True).to_dict()
    }

conditional_fits = {}
for column in numerical.columns:
    anomaly_data = train_data[train_data['class'] == 'anomaly']
    normal_data = train_data[train_data['class'] == 'normal']
    
    filtered_anomaly_col = anomaly_data[column][anomaly_data[column] < anomaly_data[column].quantile(0.99)]
    filtered_normal_col = normal_data[column][normal_data[column] < normal_data[column].quantile(0.99)]
    
    if column in best_fits:
        best_dist_name = best_fits[column]['Best Distribution']
        best_dist = distributions[best_dist_name]
        
        anomaly_params = best_dist.fit(filtered_anomaly_col)
        normal_params = best_dist.fit(filtered_normal_col)
        
        conditional_fits[column] = {
            'Anomaly': {'Parameters': anomaly_params, 'Distribution': best_dist_name},
            'Normal': {'Parameters': normal_params, 'Distribution': best_dist_name}
        }

def naive_bayes_probability(row, conditional_fits, pmf_fits, true_prob):
    anomaly_prob = true_prob['Anomaly']
    normal_prob = true_prob['Normal']
    
    for column in conditional_fits:
        value = row[column]
        dist_name = conditional_fits[column]['Anomaly']['Distribution']
        dist = distributions[dist_name]
        
        params_anomaly = conditional_fits[column]['Anomaly']['Parameters']
        params_normal = conditional_fits[column]['Normal']['Parameters']
        
        pdf_anomaly = dist.pdf(value, *params_anomaly)
        pdf_normal = dist.pdf(value, *params_normal)
        
        anomaly_prob *= pdf_anomaly if pdf_anomaly > 0 else 1e-10
        normal_prob *= pdf_normal if pdf_normal > 0 else 1e-10
    
    for column in pmf_fits:
        value = row[column]
        anomaly_prob *= pmf_fits[column]['Anomaly'].get(value, 1e-10)
        normal_prob *= pmf_fits[column]['Normal'].get(value, 1e-10)
    
    total = anomaly_prob + normal_prob
    return anomaly_prob / total


num_anomalies = train_data['class'].value_counts().get('anomaly', 0)
num_normals = train_data['class'].value_counts().get('normal', 0)
total_samples = num_anomalies + num_normals

true_probabilities = {
    'Anomaly': num_anomalies / total_samples,
    'Normal': num_normals / total_samples
}

# Evaluate on test dataset
predictions = []
true_labels = test_anomaly_class.values
for _, row in test_data.iterrows():
    anomaly_probability = naive_bayes_probability(row, conditional_fits, pmf_fits, true_probabilities)
    predictions.append(1 if anomaly_probability > 0.01 else 0)

predictions = np.array(predictions)
true_labels = np.array(true_labels)

TP = ((predictions == 1) & (true_labels == 1)).sum()
TN = ((predictions == 0) & (true_labels == 0)).sum()
FP = ((predictions == 1) & (true_labels == 0)).sum()
FN = ((predictions == 0) & (true_labels == 1)).sum()

accuracy = (TP + TN) / len(true_labels)
precision = TP / (TP + FP)
recall = TP / (TP + FN) 

print("Metrics using Naive Bayes Estimation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Task 2
categorical = categorical.drop(columns=['class'])
dfClass = df['class'].apply(lambda x: 1 if x == 'anomaly' else 0)
train_data['class'] = train_data['class'].apply(lambda x: 1 if x == 'anomaly' else 0)
test_data['class'] = test_data['class'].apply(lambda x: 1 if x == 'anomaly' else 0)

# One-Hot encoding
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(train_data[categorical.columns]) 
train_one_hot = encoder.transform(train_data[categorical.columns])
test_one_hot = encoder.transform(test_data[categorical.columns])
train_one_hot_df = pd.DataFrame(train_one_hot, columns=encoder.get_feature_names_out(categorical.columns))
test_one_hot_df = pd.DataFrame(test_one_hot, columns=encoder.get_feature_names_out(categorical.columns))
train_data_reset = train_data.drop(columns=categorical.columns).reset_index(drop=True)
train_one_hot_reset = pd.DataFrame(train_one_hot, columns=encoder.get_feature_names_out(categorical.columns)).reset_index(drop=True)
train_encoded = pd.concat([train_data_reset, train_one_hot_reset], axis=1)
test_data_reset = test_data.drop(columns=categorical.columns).reset_index(drop=True)
test_one_hot_reset = pd.DataFrame(test_one_hot, columns=encoder.get_feature_names_out(categorical.columns)).reset_index(drop=True)
test_encoded = pd.concat([test_data_reset, test_one_hot_reset], axis=1)
train_encoded, test_encoded = train_encoded.align(test_encoded, join="left", axis=1)

X_train = train_encoded.drop(columns=['class'])
y_train = dfClass[:split_index]
X_test = test_encoded.drop(columns=['class'])
y_test = dfClass[split_index:]

def calculate_metrics(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()

    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return accuracy, precision, recall

# GaussianNB
gnb = GaussianNB()
ypredgnb = gnb.fit(X_train, y_train).predict(X_test)
accuracy_gnb, precision_gnb, recall_gnb = calculate_metrics(y_test.values, ypredgnb)

# MultinomialNB
mnb = MultinomialNB()
ypredmnb = mnb.fit(X_train, y_train).predict(X_test)
accuracy_mnb, precision_mnb, recall_mnb = calculate_metrics(y_test.values, ypredmnb)

# BernoulliNB
bnb = BernoulliNB()
ypredbnb = bnb.fit(X_train, y_train).predict(X_test)
accuracy_bnb, precision_bnb, recall_bnb = calculate_metrics(y_test.values, ypredbnb)

print("\nGaussian Naive Bayes:")
print(f"Accuracy: {accuracy_gnb:.4f}, Precision: {precision_gnb:.4f}, Recall: {recall_gnb:.4f}")
print("\nMultinomial Naive Bayes:")
print(f"Accuracy: {accuracy_mnb:.4f}, Precision: {precision_mnb:.4f}, Recall: {recall_mnb:.4f}")
print("\nBernoulli Naive Bayes:")
print(f"Accuracy: {accuracy_bnb:.4f}, Precision: {precision_bnb:.4f}, Recall: {recall_bnb:.4f}")

# the model providing the best results is the Bernoulli Naive Bayes, taht's because it provides
# the best accuracy, percision and recall(specially recall as it is the most important metric in the project)
