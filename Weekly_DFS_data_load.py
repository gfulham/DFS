# %%
import pandas as pd
import re
import datetime
from datetime import datetime, timedelta
import numpy as np

# Clean DK names by removing suffixes
def clean_name(name):
    if pd.isna(name):
        return name
    
    # Convert to string and normalize
    cleaned = str(name).strip()
    
    # Remove common suffixes (case insensitive)
    suffixes = r'\b(Jr\.?|Sr\.?|III|II|IV|V)\b'
    cleaned = re.sub(suffixes, '', cleaned, flags=re.IGNORECASE)

    # Handle common apostrophe variations (normalize them all)
    # Replace curly apostrophes, straight quotes, etc. with standard apostrophe
    cleaned = re.sub(r"[''`]", "'", cleaned)
    
    # Remove all punctuation EXCEPT apostrophes (keep De'Von as DeVon after next step)
    cleaned = re.sub(r"[^A-Za-z' ]+", '', cleaned)
    
    # Now remove apostrophes to standardize (De'Von -> DeVon)
    cleaned = cleaned.replace("'", "")
    
    # Collapse multiple spaces and trim
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Convert to lowercase for consistent matching
    cleaned = cleaned.lower()
    
    return cleaned


# Clean the position column to remove /Flex
def clean_position(pos):
    if pd.isna(pos):
        return pos
    return pos.replace('/Flex', '')


def load_weekly_data(day):
    DK_salaries = pd.read_csv(f"C:\\Users\\onlyu\\OneDrive\\Fantasy Football\\data\\DKSalaries_{day}.csv")
    Vegas_projection = pd.read_csv(f"C:\\Users\\onlyu\\OneDrive\\Fantasy Football\\data\\vegas_ppr_proj_{day}.csv", encoding='cp1252')
    #FD_salaries = pd.read_csv(f"C:\\Users\\onlyu\\OneDrive\\Fantasy Football\\data\\FDSalaries_{day}.csv")
    #Ownership_projections = pd.read_csv(f"C:\\Users\\onlyu\\OneDrive\\Fantasy Football\\data\\rotowire_ownership_projection_{day}.csv")
    FFA_projections = pd.read_csv(f"C:\\Users\\onlyu\\OneDrive\\Fantasy Football\\data\\FFA_Projections_{day}.csv")
    #Injury_report = pd.read_csv(f"C:\\Users\\onlyu\\OneDrive\\Fantasy Football\\data\\Injury_Report_{day}.csv", encoding='cp1252')
    
    # %%
    # DST Comes from FFA Projections since vegas does not have lines for defenses
    DST_projections = FFA_projections[FFA_projections['position'] == 'DST']
    DST_projections = DST_projections[['player', 'points']]
    DST_projections = DST_projections.rename(columns={'points': 'Projection', 'player': 'Name'})
    # display(DST_projections)

    # %%
    DST_projections = FFA_projections[FFA_projections['position'] == 'DST']
    DST_projections = DST_projections[['player', 'points']]
    DST_projections = DST_projections.rename(columns={'points': 'Projection', 'player': 'Name_Clean'})
    #isplay(DST_projections)

    # %%
    Vegas_projection['Name_Clean'] = Vegas_projection['Name'].apply(clean_name)
    DK_salaries['Name_Clean'] = DK_salaries['Name'].apply(clean_name)
    DK_salaries['Roster Position'] = DK_salaries['Position'].apply(clean_position)
    FFA_projections['Name_Clean'] = FFA_projections['player'].apply(clean_name)
    #Injury_report['Name_Clean'] = Injury_report['Player'].apply(clean_name)

    # %%
    merged_df = (
        DK_salaries
        .merge(Vegas_projection[['Name_Clean', 'Projection']], left_on='Name_Clean', right_on='Name_Clean', how='left')
        .merge(DST_projections[['Name_Clean', 'Projection']], left_on='Name_Clean', right_on='Name_Clean', how='left', suffixes=('_vegas', '_dst'))
        .merge(FFA_projections[['Name_Clean', 'points']], left_on='Name_Clean', right_on='Name_Clean', how='left')
        #.merge(Injury_report[['Name_Clean', 'Injury Status']], left_on='Name_Clean', right_on='Name_Clean', how='left')
    )
    # %%
    # combine into a single Projection column (prefer player Vegas proj, fall back to DST proj)
    merged_df['Projection'] = merged_df['Projection_vegas'].fillna(merged_df['Projection_dst'])
    merged_df['Projection'] = merged_df['Projection'].fillna(merged_df['points'])

    merged_df = merged_df.drop(columns=['Projection_vegas', 'Projection_dst', 'Name_y'], errors='ignore')

    # %%

    print(f"Original DK rows: {len(DK_salaries)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Missing projections: {merged_df['Projection'].isna().sum()}")

    # Show all players from DK who don't have Vegas projections
    missing_projections = merged_df[merged_df['Projection'].isna()]

    print("Players missing projections:")
    print(missing_projections[['Name_Clean', 'Position', 'Salary']].to_string(index=False))
    #display(merged_df.sort_values(by='Projection', ascending=False).head(5))


    # %%
    # Add value calculations to the merged dataframe
    merged_df['Points_Per_Dollar'] = merged_df['Projection'] / merged_df['Salary'] * 1000  # Points per $1000

    # Add value rank within each position
    merged_df['Value_Rank'] = merged_df.groupby('Position')['Points_Per_Dollar'].rank(ascending=False)

    # Show best values overall
    best_values = merged_df[merged_df['Projection'].notna()].sort_values('Points_Per_Dollar', ascending=False)

    return merged_df, best_values
