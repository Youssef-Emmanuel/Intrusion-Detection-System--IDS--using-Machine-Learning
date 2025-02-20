import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Train_data.csv')
numerical= df.select_dtypes(include=[np.number])
categorical = df.select_dtypes(exclude=[np.number])
print(df.columns)
print(df.dtypes)
print(df.isna())
print(np.isinf(numerical))
print(df.nunique())

for i in numerical.columns:
    max = numerical[i].max()
    min = numerical[i].min()
    mean= numerical[i].mean()
    variance = numerical[i].var()
    print(f"\nColumn: {i}")
    print(f"  Maximum: {max}")
    print(f"  Minimum: {min}")
    print(f"  Mean: {mean}")
    print(f"  Variance: {variance}")
    
# Quarters    
for i in numerical:
    df[i +'_quartiles'] = pd.cut(df[i], bins=4, labels=False)
    quarter_data = df.groupby(i + '_quartiles')[i].agg(['max', 'min', 'mean', 'var'])
    print(f"\ndata of {i} in 4 quarters:\n", quarter_data)
          
#PDF AND PMF       
for i in df.columns:
    if pd.api.types.is_numeric_dtype(df[i]):
        num_unique_values = df[i].nunique()
        if num_unique_values > 30:
            plt.figure(figsize= (12,6))
            points, bin_edges = np.histogram(df[i],bins = 50, density= True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            plt.plot(bin_centers, points)
            plt.title(f"PDF for {i}")
            plt.xlabel(i)
            plt.ylabel('Density')
            plt.show()

        else:
            plt.figure(figsize= (12,6))
            points, bin_edges = np.histogram(df[i], bins=num_unique_values)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            plt.bar(bin_centers, points, width=bin_edges[1]-bin_edges[0])
            plt.title(f"PMF for {i} ")
            plt.xlabel(i)
            plt.ylabel('Counts')
            plt.grid(True)
            plt.show()
    else:
        plt.figure(figsize= (12,6))     
        vals, unique_vals = pd.factorize(df[i])
        points, bin_edges = np.histogram(vals, bins=len(unique_vals))
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.bar(bin_centers, points, width=bin_edges[1]-bin_edges[0])
        plt.xticks(bin_centers, unique_vals)
        plt.title(f"PMF for {i}")
        plt.xlabel(i)
        plt.ylabel('Counts')
        plt.grid(True)
        plt.show()
        
# CDF for nmerical an categorial
for i in numerical.columns:
    sorted_data = numerical[i].sort_values()
    cdf = np.arange(1,len(sorted_data)+1 )/len(sorted_data)
    plt.figure(figsize= (12,6))
    plt.step(sorted_data, cdf, where='post')
    plt.title(f'CDF of {i}')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.show()
for i in categorical.columns:
    values = df[i].value_counts(normalize=True).sort_index()
    cdf = values.cumsum() 
    plt.figure(figsize= (12,6))
    plt.step(cdf.index, cdf, where='post')
    plt.title(f"CDF of {i} ")
    plt.xlabel(i)
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()

# conditional PDF/PMF on anomaly 
anomaly_data = df[df['class'] == 'anomaly']
for col in df.columns:
    if col == 'class':
        continue  
    plt.figure(figsize= (12,6))
    unique_count = df[col].nunique()
    if pd.api.types.is_numeric_dtype(df[col]):
        if unique_count <=30:
            pmf = df[col].value_counts()
            categories = pmf.index
            plt.bar(categories, pmf)
            anomaly_pmf = anomaly_data[col].value_counts(normalize=True).reindex(categories, fill_value=0)
            plt.bar(categories, anomaly_pmf, color='red')
        else:
            counts, bin_edges = np.histogram(df[col], bins=50, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            plt.plot(bin_centers, counts)
            anomaly_counts, _ = np.histogram(anomaly_data[col], bins=bin_edges, density=True)
            plt.plot(bin_centers, anomaly_counts, color='red')
    else:
        pmf = df[col].value_counts(normalize=True)
        categories = pmf.index
        plt.bar(categories, pmf)
        anomaly_pmf = anomaly_data[col].value_counts(normalize=True).reindex(categories)
        plt.bar(categories, anomaly_pmf, color='red')
    plt.title(f"PMF/PDF for {col}")
    plt.xlabel(col)
    plt.ylabel('Density/Probability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Scatter graph
plt.figure(figsize= (12,6)) 
plt.scatter(df['dst_host_same_src_port_rate'], df['dst_host_srv_diff_host_rate'])
plt.title('Scatter Plot between dst_host_same_src_port_rate and dst_host_srv_diff_host_rate')
plt.xlabel('dst_host_same_src_port_rate')
plt.ylabel('dst_host_srv_diff_host_rate')
plt.grid(True)
plt.tight_layout()
plt.show()

# Joint PDF
plt.figure(figsize= (12,6))
hist, xedges, yedges = np.histogram2d(df['same_srv_rate'], df['diff_srv_rate'], density=True)
plt.imshow(hist.T, origin='lower', cmap='Blues', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar(label='Density')
plt.xlabel('same_srv_rate')
plt.ylabel('diff_srv_rate')
plt.grid(False)
plt.tight_layout()
plt.show()

# Condiyional joint PDF on anomaly
anomaly_df = df[df['class'] == 'anomaly']
hist, xedges, yedges = np.histogram2d(df['count'], df['srv_count'], bins=50, density=True)
hist_anomaly, xedges_anomaly, yedges_anomaly = np.histogram2d(anomaly_df['count'], anomaly_df['srv_count'], bins=50, density=True)
plt.figure(figsize= (12,6))
plt.subplot(1, 2, 1)
plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', aspect='auto', cmap='Blues')
plt.xlim(0, 300)
plt.ylim(0, 50)
plt.colorbar(label='Density')
plt.xlabel('count')
plt.ylabel('srv_count')
plt.title('Joint PDF of count and srv_count (All Data)')
plt.subplot(1, 2, 2)
plt.imshow(hist_anomaly.T, extent=[xedges_anomaly[0], xedges_anomaly[-1], yedges_anomaly[0], yedges_anomaly[-1]], origin='lower', aspect='auto', cmap='Reds')
plt.xlim(0, 300)
plt.ylim(0, 50)
plt.colorbar(label='Density')
plt.xlabel('count')
plt.ylabel('srv_count')
plt.title('Joint PDF of count and srv_count (Anomaly Data)')
plt.tight_layout()
plt.show()

#Correlation
correlation= numerical.corr()
print(correlation)
plt.figure(figsize= (12,6))
plt.imshow(correlation, cmap='coolwarm')
plt.colorbar(label='Correlation Coefficient')
plt.title('Correlation')
plt.grid(False)
plt.show()

# Dependant fields using their mean according to the class column
attack_column = 'class'
for i in numerical.columns:
    mean = df.groupby(attack_column)[i].mean()
    plt.figure(figsize=(12,6))
    mean.plot(kind='bar')
    plt.title(f'Mean of {i} by {attack_column}')
    plt.xlabel(attack_column)
    plt.ylabel(f'Mean {i}')
    plt.tight_layout()
    plt.show()









