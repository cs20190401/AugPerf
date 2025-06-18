import numpy as np
from pathlib import Path
from matchmaker import Matchmaker
from matchmaker.utils.misc import plot_and_save_score_following_result

# Load saved data
data = np.load("./dtw_results/dtw_data_online_dtw.npz", allow_pickle=True)
wp = data["wp"]
ref_features = data["ref_features"]
input_features = data["input_features"]
score_annots = data["score_annots"]
perf_annots = data["perf_annots"]
frame_rate = 22050 // 441  # Use your BEATNET_SR // CHUNK value

save_dir = Path("./dtw_results")
save_dir.mkdir(exist_ok=True)

plot_and_save_score_following_result(
    wp,
    ref_features,
    input_features,
    "cityblock",
    save_dir,
    score_annots,
    perf_annots,
    frame_rate,
    name="online_dtw"
)
print("Visualization saved to", save_dir)