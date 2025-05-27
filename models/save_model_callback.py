import shutil
from transformers import TrainerCallback


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, dest_dir):
        self.dest_dir = dest_dir

    def on_train_end(self, args, state, control, **kwargs):
        best_ckpt = state.best_model_checkpoint
        if best_ckpt is None:
            print("No best checkpoint found to copy.")
        else:
            print(f"📦 Copying best model from {best_ckpt} to {self.dest_dir}")
            shutil.copytree(best_ckpt, self.dest_dir, dirs_exist_ok=True)
