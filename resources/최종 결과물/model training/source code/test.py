from allin1.training.evaluate import evaluate
from tqdm import tqdm

project_name = "hyundai_final"
save_dir = "result"

RUN_ID = [
  # type your run_id
  # ex) "y9ey51pw", "ejr85vjh"
]

for run_id in tqdm(RUN_ID):
    print(f'=> Running evaluation of {run_id}...')
    evaluate(run_id=run_id, project_name=project_name, save_dir=save_dir)