from .actions import actions_from_poses, preprocess_image, ActionNormalizer, ACTION_STRIDE, N_FUTURE


def __getattr__(name):          # lazy: importing droid_wm.actions must not pull in torch/diffsynth
    if name == "DroidWorldModel":
        from .model import DroidWorldModel
        return DroidWorldModel
    raise AttributeError(name)
