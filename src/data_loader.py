
import pandas as pd

def load_and_merge():
    # Load raw data
    df1 = pd.read_csv('Audible_Catlog.csv')
    df2 = pd.read_csv('Audible_Catlog_Advanced_Features.csv')
    
    # Merge on 'Book Name' and 'Author'
    # We use 'left' merge to keep the 6,368 books from the main catalog 
    # and attach descriptions where available.
    df = pd.merge(df1, df2, on=['Book Name', 'Author'], how='left', suffixes=('', '_extra'))
    
    # Clean up duplicate columns from merge
    cols_to_drop = [c for c in df.columns if '_extra' in c]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Clean Genre: Extracting the specific category from the messy string
    def extract_genre(text):
        if pd.isna(text): return "General"
        return text.split(',')[-1].strip().replace('#', '')
    
    df['Primary_Genre'] = df['Ranks and Genre'].apply(extract_genre)
    
    # Handle missing values for ratings and reviews
    df['Number of Reviews'] = df['Number of Reviews'].fillna(0)
    df['Rating'] = df['Rating'].replace(-1, 0) # Some rows have -1 for no rating
    
    return df