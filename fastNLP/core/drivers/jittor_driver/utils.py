from contextlib import ExitStack

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    import jittor

class DummyGradScaler:
    """
    用于仿造的GradScaler对象，防止重复写大量的if判断

    """
    def __init__(self, *args, **kwargs):
        pass

    def get_scale(self):
        return 1.0

    def is_enabled(self):
        return False

    def scale(self, outputs):
        return outputs

    def step(self, optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        pass

    def unscale_(self, optimizer):
        pass

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}


def _build_fp16_env(dummy=False):
    if dummy:
        auto_cast = ExitStack
        GradScaler = DummyGradScaler
    else:
        raise NotImplementedError("JittorDriver does not support fp16 now.")
        # if not jt.flags.use_cuda:
        #     raise RuntimeError("No cuda")
        # if paddle.device.cuda.get_device_capability(0)[0] < 7:
        #     log.warning(
        #         "NOTE: your device does NOT support faster training with fp16, "
        #         "please switch to FP32 which is likely to be faster"
        #     )
        # from paddle.amp import auto_cast, GradScaler
    return auto_cast, GradScaler