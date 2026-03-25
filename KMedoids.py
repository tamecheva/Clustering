import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import cdist_dtw
from sklearn.manifold import MDS
from scipy.interpolate import griddata
from sklearn.metrics import silhouette_score

# --- Пътища ---
base_folder = r"C:\Users\Teodora Mecheva\Documents\Учебни часове\paper2"
result_folder = os.path.join(base_folder, "Result")
os.makedirs(result_folder, exist_ok=True)

years = ["2022", "2023", "2024"]
all_df = pd.DataFrame()

# --- Четене ---
for year in years:
    csv_path = os.path.join(result_folder, f"{year}.csv")
    if os.path.exists(csv_path):
        df_year = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
        df_year['v_count'] = pd.to_numeric(df_year['v_count'], errors='coerce')
        df_year['v_vel']   = pd.to_numeric(df_year['v_vel'], errors='coerce')

        df_year = df_year.groupby(['date', 'Direction'], as_index=False).agg({
            'v_count': 'sum',
            'v_vel': 'mean'
        })

        all_df = pd.concat([all_df, df_year], ignore_index=True)

if all_df.empty:
    raise ValueError("Няма данни.")

# --- Clean ---
all_df['Direction'] = all_df['Direction'].astype(str).str.strip()
all_df['Direction'] = all_df['Direction'].replace({
    'Околовръстно': 'Ring Road',
    'Център': 'Center'
})

# --- Дата ---
all_df['date'] = pd.to_datetime(all_df['date'], errors='coerce')
all_df['Hour'] = all_df['date'].dt.hour.fillna(0).astype(int)
all_df['DateOnly'] = pd.to_datetime(all_df['date'].dt.date)

# --- Pivot ---
df_pivot_count = all_df.pivot_table(index='DateOnly', columns=['Direction','Hour'], values='v_count')
df_pivot_vel   = all_df.pivot_table(index='DateOnly', columns=['Direction','Hour'], values='v_vel')

# --- Пълни часове ---
full_hours = list(range(24))
full_columns = pd.MultiIndex.from_product([all_df['Direction'].unique(), full_hours])
df_pivot_count = df_pivot_count.reindex(columns=full_columns)
df_pivot_vel   = df_pivot_vel.reindex(columns=full_columns)

# =========================================================
# ✅ SAVE RAW MATRICES (ПРЕДИ ИНТЕРПОЛАЦИЯ)
# =========================================================
raw_folder = os.path.join(result_folder, "Raw_24h_matrices")
os.makedirs(raw_folder, exist_ok=True)

def flatten_columns(df):
    df_copy = df.copy()
    df_copy.columns = [f"{d}_{h:02d}" for d, h in df_copy.columns]
    return df_copy

df_count_raw = flatten_columns(df_pivot_count)
df_count_raw.to_csv(os.path.join(raw_folder, "count_24h_raw.csv"), encoding='utf-8-sig')

df_vel_raw = flatten_columns(df_pivot_vel)
df_vel_raw.to_csv(os.path.join(raw_folder, "velocity_24h_raw.csv"), encoding='utf-8-sig')

print("✅ Raw 24h matrices saved (before interpolation)")

# --- Fill с 2D интерполация ---
def fill_2d_interpolation(df):
    df_filled = df.copy()
    rows, cols = np.meshgrid(np.arange(df.shape[0]), np.arange(df.shape[1]), indexing='ij')
    mask = df_filled.isna().values
    points = np.array([rows[~mask], cols[~mask]]).T
    values = df_filled.values[~mask]
    missing_points = np.array([rows[mask], cols[mask]]).T
    if len(missing_points) > 0:
        interpolated = griddata(points, values, missing_points, method='linear')
        df_filled.values[mask] = interpolated
        df_filled = df_filled.apply(lambda col: col.fillna(col.median()))
    return df_filled

df_pivot_count = fill_2d_interpolation(df_pivot_count)
df_pivot_vel   = fill_2d_interpolation(df_pivot_vel)

# --- Split ---
def select_cols(df, direction):
    return [col for col in df.columns if col[0] == direction]

data_dict = {
    'Ring Count': df_pivot_count[select_cols(df_pivot_count,'Ring Road')],
    'Center Count': df_pivot_count[select_cols(df_pivot_count,'Center')],
    'Ring Vel': df_pivot_vel[select_cols(df_pivot_vel,'Ring Road')],
    'Center Vel': df_pivot_vel[select_cols(df_pivot_vel,'Center')]
}

# --- Подготовка ---
def prepare_data(df):
    X = df.values.astype(float)
    idx = df.index
    return X, idx

clean_data_dict = {}
for key, df in data_dict.items():
    if df.shape[0] == 0 or df.shape[1] == 0:
        continue
    X, idx = prepare_data(df)
    if X.shape[0] > 200:
        X = X[::2]
        idx = idx[::2]
    clean_data_dict[key] = (X, idx)

# --- K-Medoids ---
def kmedoids_from_distance(D, k, max_iter=100):
    np.random.seed(42)
    medoids = np.random.choice(len(D), k, replace=False)
    labels = np.zeros(len(D), dtype=int)

    for _ in range(max_iter):
        for i in range(len(D)):
            labels[i] = np.argmin([D[i, m] for m in medoids])

        new_medoids = []
        for i in range(k):
            idx = np.where(labels == i)[0]
            if len(idx) == 0:
                new_medoids.append(np.random.randint(len(D)))
                continue
            subD = D[np.ix_(idx, idx)]
            new_medoids.append(idx[np.argmin(subD.sum(axis=1))])

        new_medoids = np.array(new_medoids)
        if np.all(medoids == new_medoids):
            break
        medoids = new_medoids

    return labels, medoids

# --- Избор на k ---
def find_best_k(X, name):
    X_3d = X.reshape(X.shape[0], X.shape[1], 1)
    D = cdist_dtw(X_3d, global_constraint="sakoe_chiba", sakoe_chiba_radius=2)

    k_range = range(2, 6)
    scores = []

    best_score = -1
    best_k = 2
    best_labels = None
    best_medoids = None

    for k in k_range:
        labels, medoids = kmedoids_from_distance(D, k)
        try:
            score = silhouette_score(D, labels, metric='precomputed')
        except:
            score = -1

        scores.append(score)

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_medoids = medoids

    # Plot
    plt.figure()
    plt.plot(list(k_range), scores, marker='o')
    plt.title(f"Silhouette vs k - {name}")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.grid()

    sil_folder = os.path.join(result_folder, name.split()[0], 'Silhouette')
    os.makedirs(sil_folder, exist_ok=True)
    plt.savefig(os.path.join(sil_folder, f"{name}_silhouette.png"))
    plt.close()

    print(f"{name}: best k = {best_k}, score={best_score:.4f}")

    return best_k, best_labels, best_medoids, D

# --- Visualization ---
def visualize(X, labels, medoids, D, name):
    direction = name.split()[0]
    type_val = name.split()[1]

    folder_base = os.path.join(result_folder, direction, type_val)
    os.makedirs(folder_base, exist_ok=True)

    # MDS
    from mpl_toolkits.mplot3d import Axes3D
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    X_mds = mds.fit_transform(D)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in np.unique(labels):
        idx = labels == i
        ax.scatter(X_mds[idx,0], X_mds[idx,1], X_mds[idx,2], label=f'C{i}', alpha=0.7)

    for i, m in enumerate(medoids):
        ax.scatter(X_mds[m,0], X_mds[m,1], X_mds[m,2], marker='x', s=200)

    ax.set_title(f"MDS 3D - {name}")
    ax.legend()

    plt.savefig(os.path.join(folder_base, f"{name}_mds.png"))
    plt.close()

    # Medoids
    plt.figure(figsize=(10,5))
    for i, m in enumerate(medoids):
        plt.plot(X[m], label=f'Cluster {i}')

    plt.title(f"Medoids - {name}")
    plt.xlabel("Hour index")
    plt.ylabel("Absolute value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(folder_base, f"{name}_medoids.png"))
    plt.close()

# --- RUN ---
for name, (X, idx) in clean_data_dict.items():
    best_k, labels, medoids, D = find_best_k(X, name)
    visualize(X, labels, medoids, D, name)

print("\n✅ Готово!") 