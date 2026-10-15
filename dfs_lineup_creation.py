# %%
import pandas as pd
import re
import datetime
from datetime import datetime, timedelta
import numpy as np
#head to view all rows
pd.options.display.max_rows = 1000
# Clean DK names by removing suffixes
def clean_name(name):
    if pd.isna(name):
        return name
    
    # Remove common suffixes (case insensitive)
    suffixes = r'\b(Jr\.?|Sr\.?|III|II|IV|V)\b'
    cleaned = re.sub(suffixes, '', str(name), flags=re.IGNORECASE)

    # Remove all punctuation (anything not letter/number/space), replace with space
    #cleaned = re.sub(r'[^A-Za-z0-9 ]+', ' ', cleaned)
    # If you want   ONLY letters (no digits), uncomment the next line:
    cleaned = re.sub(r'[^A-Za-z ]+', ' ', cleaned)
    # Collapse multiple spaces and trim
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned
# Get todays date in YYYY_MM_DD

today = datetime.today()
print(f"Today's date is: {today}")

#This Sunday
Sunday = today - timedelta(days=(datetime.now().weekday() + 1) % 7) 
Sunday = Sunday.strftime('%Y_%m_%d')
print(f"Last Sunday's date was: {Sunday}")

# Clean the position column to remove /Flex
def clean_position(pos):
    if pd.isna(pos):
        return pos
    return pos.replace('/Flex', '')

# %%
DK_salaries = pd.read_csv(f"C:\\Users\\onlyu\\OneDrive\\Fantasy Football\\data\\DKSalaries_{Sunday}.csv")
Vegas_projection = pd.read_csv(f"C:\\Users\\onlyu\\OneDrive\\Fantasy Football\\data\\vegas_ppr_proj_{Sunday}.csv")
FD_salaries = pd.read_csv(f"C:\\Users\\onlyu\\OneDrive\\Fantasy Football\\data\\FDSalaries_{Sunday}.csv")
Ownership_projections = pd.read_csv(f"C:\\Users\\onlyu\\OneDrive\\Fantasy Football\\data\\rotowire_ownership_projection_{Sunday}.csv")
FFA_projections = pd.read_csv(f"C:\\Users\\onlyu\\OneDrive\\Fantasy Football\\data\\FFA_Projections_{Sunday}.csv")



# %% [markdown]
# DST Comes from FFA Projections since vegas does not have lines for defenses

# %%
DST_projections = FFA_projections[FFA_projections['position'] == 'DST']
DST_projections = DST_projections[['player', 'points']]
DST_projections = DST_projections.rename(columns={'points': 'Projection', 'player': 'Name'})
display(DST_projections)

# %%
Vegas_projection['Name_Clean'] = Vegas_projection['Name'].apply(clean_name)
DK_salaries['Name_Clean'] = DK_salaries['Name'].apply(clean_name)
DK_salaries['Roster Position'] = DK_salaries['Position'].apply(clean_position)

display(DK_salaries.head(1))
display(Vegas_projection.head(1))
# Left join DK salaries with Vegas projections on name/player


# %%

merged_df = (
    DK_salaries
    .merge(Vegas_projection[['Name_Clean', 'Projection']], left_on='Name_Clean', right_on='Name_Clean', how='left')
    .merge(DST_projections, left_on='Name_Clean', right_on='Name', how='left')
)

display(merged_df.head(1))

# %%

# combine into a single Projection column (prefer player Vegas proj, fall back to DST proj)
merged_df['Projection'] = merged_df['Projection_x'].fillna(merged_df['Projection_y'])


merged_df = merged_df.drop(columns=['Projection_x', 'Projection_y', 'Name_y'])


# %%

print(f"Original DK rows: {len(DK_salaries)}")
print(f"Merged rows: {len(merged_df)}")
print(f"Missing projections: {merged_df['Projection'].isna().sum()}")

# Show all players from DK who don't have Vegas projections
missing_projections = merged_df[merged_df['Projection'].isna()]

print("Players missing projections:")
print(missing_projections[['Name_Clean', 'Position', 'Salary']].to_string(index=False))
display(merged_df.sort_values(by='Projection', ascending=False).head(5))


# %%
# Add value calculations to the merged dataframe
merged_df['Points_Per_Dollar'] = merged_df['Projection'] / merged_df['Salary'] * 1000  # Points per $1000

# Add value rank within each position
merged_df['Value_Rank'] = merged_df.groupby('Position')['Points_Per_Dollar'].rank(ascending=False)

# Show best values overall
best_values = merged_df[merged_df['Projection'].notna()].sort_values('Points_Per_Dollar', ascending=False)

print("Best values (Points per $1000):")
print(best_values[['Name_Clean', 'Position', 'Salary', 'Projection', 'Points_Per_Dollar']].head(10).to_string(index=False))

# Show best values by position
print("\nBest values by position:")
for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
    pos_values = best_values[best_values['Position'] == pos].head(10)
    if not pos_values.empty:
        print(f"\n{pos}:")
        print(pos_values[['Name_Clean', 'Salary', 'Projection', 'Points_Per_Dollar']].to_string(index=False))

# %% [markdown]
# ## Old lineup creation of player combinations
# 
# Using for loops is very computational intensive. Working on other solutions to speed up lineup creation

# %% [markdown]
# ## New lineup creation using numpy

# %%
# ...existing code...
import numpy as np
from itertools import combinations
import math

# If we want to see a lot of print statements
VERBOSE = True

# Filter out players without projections and sort by value
valid_players = merged_df[merged_df['Projection'].notna()].copy()

# Get top players by position (expand pools since we need more)
qbs = valid_players[valid_players['Position'] == 'QB'].nlargest(10, 'Points_Per_Dollar')
rbs = valid_players[valid_players['Position'] == 'RB'].nlargest(20, 'Points_Per_Dollar')
wrs = valid_players[valid_players['Position'] == 'WR'].nlargest(20, 'Points_Per_Dollar')
tes = valid_players[valid_players['Position'] == 'TE'].nlargest(10, 'Points_Per_Dollar')
dsts = valid_players[valid_players['Position'] == 'DST'].nlargest(10, 'Points_Per_Dollar')

#Total combinations
total_combos = (len(qbs) * math.comb(len(rbs), 2) *
                math.comb(len(wrs), 3) * len(tes) * len(dsts) *
                (len(rbs) - 2 + len(wrs) - 3 + len(tes)))  # FLEX options

print(f"Total possible lineups with expanded pools: {total_combos}")

max_salary = 50000

# Convert pools to NumPy
def arrs(df):
    return (
        df["Salary"].to_numpy(np.int32),
        df["Projection"].to_numpy(np.float32),
        df["Name_Clean"].to_numpy(object),
    )

# Pools as arrays
qb_sal, qb_pts, qb_names   = arrs(qbs)
rb_sal, rb_pts, rb_names   = arrs(rbs)
wr_sal, wr_pts, wr_names   = arrs(wrs)
te_sal, te_pts, te_names   = arrs(tes)
dst_sal, dst_pts, dst_names= arrs(dsts)

# Number of players per position
n_qb, n_rb, n_wr, n_te, n_dst = len(qb_sal), len(rb_sal), len(wr_sal), len(te_sal), len(dst_sal)

if VERBOSE:
    from pprint import pprint

    def show_pool(tag, sal, pts, names, k=5):
        print(f"[{tag}] n={len(sal)} | salary dtype={sal.dtype}, range=({sal.min()}-{sal.max()}) "
        f"| pts dtype={pts.dtype}, range=({pts.min():.2f}-{pts.max():.2f})")
        print("    sample names:", end=" "); pprint(list(names[:k]))

    print("Pool sizes (QB, RB, WR, TE, DST):", (n_qb, n_rb, n_wr, n_te, n_dst))
    show_pool("QB",  qb_sal, qb_pts, qb_names)
    show_pool("RB",  rb_sal, rb_pts, rb_names)
    show_pool("WR",  wr_sal, wr_pts, wr_names)
    show_pool("TE",  te_sal, te_pts, te_names)
    show_pool("DST", dst_sal, dst_pts, dst_names)

    base_total = n_qb * (n_rb*(n_rb-1)//2) * (n_wr*(n_wr-1)*(n_wr-2)//6) * n_te * n_dst
    print(f"Estimated base combos (Q × C(RB,2) × C(WR,3) × TE × DST): {base_total:,}")
# ...existing code...


# %%
# Precompute RB pairs and WR trios
# Because we need to select 2 RBs and 3 WRs, we can precompute all combinations
# and their combined salaries and points to speed up lineup generation
rb_pairs = np.array(list(combinations(np.arange(n_rb, dtype=np.int16), 2)), dtype=np.int16)  # (Crb,2)
wr_trios = np.array(list(combinations(np.arange(n_wr, dtype=np.int16), 3)), dtype=np.int16) # (Cwr,3)

Crb, Cwr = len(rb_pairs), len(wr_trios)

rb_pair_sal = rb_sal[rb_pairs].sum(axis=1).astype(np.int32)
rb_pair_pts = rb_pts[rb_pairs].sum(axis=1).astype(np.float32)

wr_trio_sal = wr_sal[wr_trios].sum(axis=1).astype(np.int32)
wr_trio_pts = wr_pts[wr_trios].sum(axis=1).astype(np.float32)


# ...existing code...
if VERBOSE:
    print(f"RB pairs: {len(rb_pairs):,}, WR trios: {len(wr_trios):,}")
    print("First 3 RB pairs (idx):", rb_pairs[:3].tolist())
    print("First 2 WR trios (idx):", wr_trios[:2].tolist())
    print("First 3 RB pair names:", [(rb_names[i], rb_names[j]) for i,j in rb_pairs[:3]])
    print("First 2 WR trio names:", [(wr_names[i], wr_names[j], wr_names[k]) for i,j,k in wr_trios[:2]])
# 


# %%

# Build FLEX pool (RB + WR + TE)
flex_salary = np.concatenate([rb_sal, wr_sal, te_sal]).astype(np.int32)
flex_points = np.concatenate([rb_pts, wr_pts, te_pts]).astype(np.float32)
flex_names  = np.concatenate([rb_names, wr_names, te_names])

# number of FLEX candidates
n_flex = len(flex_salary)

# Is the flex position an RB, WR, or TE?
# FLEX position/type flags and local indices
# Start with np.zeros and set True for each position
# The n_flex is built from flex_salary which is concatenated from RB, WR, TE, so we can slice accordingly
flex_is_rb = np.zeros(n_flex, dtype=bool); flex_is_rb[:n_rb] = True
flex_is_wr = np.zeros(n_flex, dtype=bool); flex_is_wr[n_rb:n_rb+n_wr] = True
flex_is_te = np.zeros(n_flex, dtype=bool); flex_is_te[n_rb+n_wr:] = True

# We do this beacuse we need to know which player in the FLEX pool corresponds to which RB/WR/TE index 
# Fill th idx arrays with -1 
flex_rb_idx = np.full(n_flex, -1, dtype=np.int16); flex_rb_idx[:n_rb] = np.arange(n_rb, dtype=np.int16)
flex_wr_idx = np.full(n_flex, -1, dtype=np.int16); flex_wr_idx[n_rb:n_rb+n_wr] = np.arange(n_wr, dtype=np.int16)
flex_te_idx = np.full(n_flex, -1, dtype=np.int16); flex_te_idx[n_rb+n_wr:] = np.arange(n_te, dtype=np.int16)

if VERBOSE:
    print("flex_rb_idx:", flex_rb_idx)  # [ 0  1  2  3  4  5  6  7  8  9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]    
    print("flex_wr_idx:", flex_wr_idx)  # [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0  1  2  3  4  5  6  7  8  9 -1 -1 -1 -1 -1]
    print("flex_te_idx:", flex_te_idx)  # [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0  1  2  3  4]
    
    # Example: we already chose RB indices (0,1). Which FLEX slots are safe for RB duplication?
    chosen_rb_pair = np.array([0, 1], dtype=np.int16)
    is_duplicate_rb = np.isin(flex_rb_idx, chosen_rb_pair)  # True where FLEX is RB0 or RB1
    print("dup RB flags:", is_duplicate_rb)  # [ True  True False False False False]
    


# %%

# Precompute FLEX overlap masks versus RB-pairs / WR-trios / TE
# Shapes: (Crb, n_flex), (Cwr, n_flex), (n_te, n_flex)
# rb_flex_ok equals True where FLEX can be that position without duplication of already chosen RBs/WRs/TE
# This works by broadcasting the flex_idx arrays against the rb_pairs and wr_trios
rb_flex_ok = (~flex_is_rb)[None, :] | (
    (flex_rb_idx[None, :] != rb_pairs[:, None, 0]) &
    (flex_rb_idx[None, :] != rb_pairs[:, None, 1])
)

wr_flex_ok = (~flex_is_wr)[None, :] | (
    (flex_wr_idx[None, :] != wr_trios[:, None, 0]) &
    (flex_wr_idx[None, :] != wr_trios[:, None, 1]) &
    (flex_wr_idx[None, :] != wr_trios[:, None, 2])
)

te_flex_ok = (~flex_is_te)[None, :] | (
    (flex_te_idx[None, :] != np.arange(n_te, dtype=np.int16)[:, None])
)

# The shape shows how many RB pairs, WR trios, and TEs we have versus the FLEX pool size
# THe flex pool size is 25 here because we have 10 RBs + 10 WRs + 5 TEs
if VERBOSE:
    print("RB-FLEX overlap mask shape:", rb_flex_ok.shape)
    print("WR-FLEX overlap mask shape:", wr_flex_ok.shape)
    print("TE-FLEX overlap mask shape:", te_flex_ok.shape)


# %%

# Build base grid (QB × RBpair × WRtrio × TE × DST)
# 
iq, irp, iwt, ite, idst = np.meshgrid(
    np.arange(n_qb, dtype=np.int32), # QBs
    np.arange(Crb, dtype=np.int32), # RB pairs
    np.arange(Cwr, dtype=np.int32), # WR trios
    np.arange(n_te, dtype=np.int32),
    np.arange(n_dst, dtype=np.int32),
    indexing='ij'
)

# %%

# Flatten base indices
# Each of these will be a 1D array where each index corresponds to a unique combination of (QB, RB pair, WR trio, TE, DST)
iq = iq.ravel(); irp = irp.ravel(); iwt = iwt.ravel(); ite = ite.ravel(); idst = idst.ravel()


# %%

# Compute base salaries and points
base_salary = (
    qb_sal[iq] + rb_pair_sal[irp] + wr_trio_sal[iwt] + te_sal[ite] + dst_sal[idst]
).astype(np.int32)

# Compute base points
base_points = (
    qb_pts[iq] + rb_pair_pts[irp] + wr_trio_pts[iwt] + te_pts[ite] + dst_pts[idst]
).astype(np.float32)

# Prune bases that cannot fit even the cheapest FLEX
min_flex = int(flex_salary.min())
can_fit_any_flex = base_salary <= (max_salary - min_flex) # base salary must leave room for cheapest FLEX

# Apply pruning
# 
iq = iq[can_fit_any_flex]; irp = irp[can_fit_any_flex]; iwt = iwt[can_fit_any_flex]
ite = ite[can_fit_any_flex]; idst = idst[can_fit_any_flex]
base_salary = base_salary[can_fit_any_flex]; base_points = base_points[can_fit_any_flex]

print(f"Base combos after pruning: {len(base_salary):,}")


# %%
# Enumerate FLEX with masks in chunks, keep only top_k to limit memory
top_k = 1_000_000  # top  k lineups based on points but not including FLEX yet
keep_idx = np.empty((0,), dtype=np.int64) # np.empty array to hold indices of kept lineups 
keep_iflex = np.empty((0,), dtype=np.int32) # np.empty array to hold indices of kept FLEX players
keep_points = np.empty((0,), dtype=np.float32)
keep_salary = np.empty((0,), dtype=np.int32)

# Example:
# keep_idx	keep_iflex	keep_points	keep_salary
# 88	    12	        143.5	    49800
# 27901	    2	        142.9	    50000
# 203	    50	        142.8	    49900

# Process in chunks to limit memory usage
chunk = 20000
for start in range(0, len(base_salary), chunk): # Chunks mean process 20,000 base lineups at a time over the 198 million
    end = min(start + chunk, len(base_salary))
    bs = base_salary[start:end]
    bp = base_points[start:end]
    _irp = irp[start:end]; _iwt = iwt[start:end]; _ite = ite[start:end]

    # Allowed FLEX by overlap
    allowed = rb_flex_ok[_irp] & wr_flex_ok[_iwt] & te_flex_ok[_ite]  # (chunk, n_flex)
    # Salary feasibility
    rem = (max_salary - bs)[:, None]
    allowed &= (flex_salary[None, :] <= rem)

    rows, cols = np.where(allowed)
    if rows.size == 0:
        continue

    pts = bp[rows] + flex_points[cols]
    sal = bs[rows] + flex_salary[cols]
    idx = (start + rows).astype(np.int64)

    # Append and retain top_k by points
    keep_idx = np.concatenate([keep_idx, idx], axis=0)
    keep_iflex = np.concatenate([keep_iflex, cols.astype(np.int32)], axis=0)
    keep_points = np.concatenate([keep_points, pts], axis=0)
    keep_salary = np.concatenate([keep_salary, sal], axis=0)

    # if keep_points.size > top_k * 2:
    #     kth = np.argpartition(keep_points, -top_k)[-top_k:]
    #     keep_idx, keep_iflex, keep_points, keep_salary = (
    #         keep_idx[kth], keep_iflex[kth], keep_points[kth], keep_salary[kth]
    #     )

#Final top_k trim
# if keep_points.size > top_k:
#     kth = np.argpartition(keep_points, -top_k)[-top_k:]
#     keep_idx, keep_iflex, keep_points, keep_salary = (
#         keep_idx[kth], keep_iflex[kth], keep_points[kth], keep_salary[kth]
#     )
    TRIM_THRESHOLD = top_k * 1.1  # keep buffer small

    if keep_points.size > TRIM_THRESHOLD:
        # Add slight noise to preserve diversity
        noise = np.random.normal(0, 0.1, keep_points.size)
        rank = keep_points + noise

        kth = np.argpartition(rank, -top_k)[-top_k:]
        keep_idx, keep_iflex, keep_points, keep_salary = (
            keep_idx[kth], keep_iflex[kth], keep_points[kth], keep_salary[kth]
        )

# Expand indices back to components for the kept rows
sel_iq  = iq[keep_idx]
sel_irp = irp[keep_idx]
sel_iwt = iwt[keep_idx]
sel_ite = ite[keep_idx]
sel_id  = idst[keep_idx]
sel_ifx = keep_iflex

# Unpack RB pair and WR trio members
rb1 = rb_pairs[sel_irp][:, 0]; rb2 = rb_pairs[sel_irp][:, 1]
wr1 = wr_trios[sel_iwt][:, 0]; wr2 = wr_trios[sel_iwt][:, 1]; wr3 = wr_trios[sel_iwt][:, 2]

# Build a lightweight lineup table (indices + totals). Convert to dicts only if needed later.
lineups_np = {
    "points": keep_points,
    "salary": keep_salary,
    "QB": sel_iq, "RB1": rb1, "RB2": rb2, "WR1": wr1, "WR2": wr2, "WR3": wr3, "TE": sel_ite, "DST": sel_id, "FLEX": sel_ifx
}

# Optional: materialize top N as dicts with names for downstream exposure logic
# 
#N_materialize = 40000
order = np.argsort(-lineups_np["points"])


def flex_name(idx):
    if idx < n_rb:
        return rb_names[idx], rb_sal[idx], rb_pts[idx], "RB"
    elif idx < n_rb + n_wr:
        j = idx - n_rb
        return wr_names[j], wr_sal[j], wr_pts[j], "WR"
    else:
        j = idx - n_rb - n_wr
        return te_names[j], te_sal[j], te_pts[j], "TE"
    
# Exposure-aware direct selection (replaces bulk materialization)
target_N = 10                  # number of lineups desired
max_exposure_pct = 0.3        # per-player cap (e.g. 40%)
max_per_player = max(1, math.floor(target_N * max_exposure_pct))

from collections import defaultdict
usage = defaultdict(int)
selected = []

for k in order:   # order already sorted by descending points
    if len(selected) >= target_N:
        break

    qb_i, rb1_i, rb2_i, wr1_i, wr2_i, wr3_i, te_i, dst_i, fx_i = (
        lineups_np["QB"][k], lineups_np["RB1"][k], lineups_np["RB2"][k],
        lineups_np["WR1"][k], lineups_np["WR2"][k], lineups_np["WR3"][k],
        lineups_np["TE"][k], lineups_np["DST"][k], lineups_np["FLEX"][k]
    )
    fx_name = flex_name(fx_i)  # (name, salary, points, position)

    candidate_players = [
        qb_names[qb_i],
        rb_names[rb1_i], rb_names[rb2_i],
        wr_names[wr1_i], wr_names[wr2_i], wr_names[wr3_i],
        te_names[te_i],
        fx_name[0],
        dst_names[dst_i],
    ]

    if any(usage[p] >= max_per_player for p in candidate_players):
        continue

    lineup = {
        "points": float(lineups_np["points"][k]),
        "salary": int(lineups_np["salary"][k]),
        "QB":  (qb_names[qb_i],  int(qb_sal[qb_i]),  float(qb_pts[qb_i])),
        "RB1": (rb_names[rb1_i], int(rb_sal[rb1_i]), float(rb_pts[rb1_i])),
        "RB2": (rb_names[rb2_i], int(rb_sal[rb2_i]), float(rb_pts[rb2_i])),
        "WR1": (wr_names[wr1_i], int(wr_sal[wr1_i]), float(wr_pts[wr1_i])),
        "WR2": (wr_names[wr2_i], int(wr_sal[wr2_i]), float(wr_pts[wr2_i])),
        "WR3": (wr_names[wr3_i], int(wr_sal[wr3_i]), float(wr_pts[wr3_i])),
        "TE":  (te_names[te_i],  int(te_sal[te_i]),  float(te_pts[te_i])),
        "FLEX": fx_name,
        "DST": (dst_names[dst_i], int(dst_sal[dst_i]), float(dst_pts[dst_i])),
    }
    selected.append(lineup)
    for p in candidate_players:
        usage[p] += 1

print(f"Selected {len(selected)} exposure-capped lineups (cap {max_exposure_pct:.0%}, max_per_player={max_per_player})")

# all_lineups = []
# for k in order:
#     qb_i, rb1_i, rb2_i, wr1_i, wr2_i, wr3_i, te_i, dst_i, fx_i = (
#         lineups_np["QB"][k], lineups_np["RB1"][k], lineups_np["RB2"][k],
#         lineups_np["WR1"][k], lineups_np["WR2"][k], lineups_np["WR3"][k],
#         lineups_np["TE"][k], lineups_np["DST"][k], lineups_np["FLEX"][k]
#     )
#     fx_name = flex_name(fx_i)
#     lineup = {
#         "points": float(lineups_np["points"][k]),
#         "salary": int(lineups_np["salary"][k]),
#         "QB":  (qb_names[qb_i],  int(qb_sal[qb_i]),  float(qb_pts[qb_i])),
#         "RB1": (rb_names[rb1_i], int(rb_sal[rb1_i]), float(rb_pts[rb1_i])),
#         "RB2": (rb_names[rb2_i], int(rb_sal[rb2_i]), float(rb_pts[rb2_i])),
#         "WR1": (wr_names[wr1_i], int(wr_sal[wr1_i]), float(wr_pts[wr1_i])),
#         "WR2": (wr_names[wr2_i], int(wr_sal[wr2_i]), float(wr_pts[wr2_i])),
#         "WR3": (wr_names[wr3_i], int(wr_sal[wr3_i]), float(wr_pts[wr3_i])),
#         "TE":  (te_names[te_i],  int(te_sal[te_i]),  float(te_pts[te_i])),
#         "FLEX": fx_name,
#         "DST": (dst_names[dst_i], int(dst_sal[dst_i]), float(dst_pts[dst_i])),
#     }
#     all_lineups.append(lineup)

# print(f"Materialized {len(all_lineups):,} top lineups (kept {len(keep_points):,} candidates in buffer)")
# # ...existing code...

# %% [markdown]
# ## Trying to reduce the lineup lookup time by using sets. Still Brute force for now. 

# %%

# pick top N lineups while enforcing max exposure
N = 20
# Max exposure per player
max_exposure_pct = .4
# Calculate max allowed lineups per player based on exposure percentage
max_per_player = max(1, math.floor(N * max_exposure_pct)) 

all_lineups = selected  # Use the previously selected lineups with exposure capping

# Initialize usage counts for all players to zero
all_lineups.sort(key=lambda x: x['points'], reverse=True) #Start with highest scoring lineups
selected = []
usage = {}  # dictionary with player name and how many lineups they are in
selected_player_sets = set() # Fast duplicate checking


def lineup_player_Name_Cleans(lu):
    Name_Cleans = [lu['QB'][0], lu['RB1'][0], lu['RB2'][0],
             lu['WR1'][0], lu['WR2'][0], lu['WR3'][0],
             lu['TE'][0], lu['FLEX'][0], lu['DST'][0]]
    return frozenset(Name_Cleans)

# Add lineup similarity checking so we don't pick identical lineups
def lineups_are_identical(lu1, lu2):
    return lineup_player_Name_Cleans(lu1) == lineup_player_Name_Cleans(lu2)

# Select lineups while enforcing max exposure and avoiding identical lineups
for i, lu in enumerate(all_lineups):
    if len(selected) >= N:
        break #end loop

    if i % 10000 == 0:
        print(f"Processed {i:,} lineups, selected {len(selected)}")


    # Fast duplicate check using frozenset
    player_names = lineup_player_Name_Cleans(lu)
    if player_names in selected_player_sets:
        continue


    # skip if any player would exceed exposure
    if any(usage.get(n, 0) >= max_per_player for n in player_names):
        continue

    # accept lineup and increment usage
    selected.append(lu)
    for n in player_names: # for each player in the lineup
        usage[n] = usage.get(n, 0) + 1 # increment player usage count
    selected_player_sets.add(player_names)

# Display selected lineups
for i, lineup in enumerate(selected, 1):
    print(f"\n--- LINEUP #{i} ---")
    print(f"Total Points: {lineup['points']:.1f} | Salary: ${lineup['salary']:,} | Remaining: ${max_salary - lineup['salary']:,}")
    print("-" * 60)
    for pos in ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']:
        player_info = lineup[pos]
        if pos == 'FLEX':
            Name_Clean, salary, points, position = player_info
            print(f"{pos:4}: {Name_Clean:25} ${salary:,} ({points:.1f} pts) [{position}]")
        else:
            Name_Clean, salary, points = player_info
            print(f"{pos:4}: {Name_Clean:25} ${salary:,} ({points:.1f} pts)")

print("\nPlayer usage counts (selected lineups):")
for player, cnt in sorted(usage.items(), key=lambda x: -x[1]):
    print(f"{player:25}: {cnt} ({cnt/len(selected):.0%} exposure)")
# ...existing code...

# %%
from collections import Counter

def lineup_names(lu):
    return [lu['QB'][0], lu['RB1'][0], lu['RB2'][0],
            lu['WR1'][0], lu['WR2'][0], lu['WR3'][0],
            lu['TE'][0], lu['FLEX'][0], lu['DST'][0]]

# Frequency across the 20k materialized lineups
freq = Counter()
for lu in all_lineups:
    freq.update(lineup_names(lu))

print("\nTop 20 player frequencies in 20k materialized:")
for name, c in freq.most_common(20):
    print(f"{name:24} {c:6}")

# Try different exposure caps
for cap in [0.35, 0.5, 0.6, 0.7]:
    max_per = max(1, math.floor(N * cap))
    usage = {}
    selected_tmp = 0
    for lu in all_lineups:
        names = lineup_names(lu)
        if any(usage.get(n,0) >= max_per for n in names):
            continue
        selected_tmp += 1
        for n in names:
            usage[n] = usage.get(n,0)+1
        if selected_tmp >= N:
            break
    print(f"Cap {cap:.2f}: max_per_player={max_per} selected={selected_tmp}")

# Count unique frozenset lineup cores (should equal len(all_lineups) unless duplicates exist)
unique_sets = len({frozenset(lineup_names(lu)) for lu in all_lineups})
print(f"\nUnique player sets in all_lineups: {unique_sets} / {len(all_lineups)}")


