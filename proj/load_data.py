import pandas as pd

def main():
    print("Step 1: Loading data...")
    country_network_df = pd.read_csv("dataset_exports_groupdescript.csv", low_memory=False)
    print('Step 1 Complete: Data loaded')
    
    test_flag = False
    if test_flag:
        print("Sampling 5% of the data for testing...")
        country_network_df = country_network_df.sample(frac=0.001, random_state=1)
        print('Step 1 Complete: Data sampled for testing')
    else:
        print("No sampling applied. Using full dataset.")
    print("Step 2: Extracting country to country data...")
    is_country_id = ~country_network_df["COUNTRY.ID"].astype(str).str.contains(r'\d', na=False)
    is_counterpart_country_id = ~country_network_df["COUNTERPART_COUNTRY.ID"].astype(str).str.contains(r'\d', na=False)
    country_to_country_network_df = country_network_df[is_country_id & is_counterpart_country_id].copy()
    print('Step 2 Complete: Country to country data extracted')

    return [
        {
            'name': 'country_to_country',
            'data': country_to_country_network_df
        }
    ]