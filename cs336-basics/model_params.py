MODEL_SMALL = 0
MODEL_MEDIUM = 1
MODEL_LARGE = 2
MODEL_XL = 3
MODEL_2700M = 4

def get_model_parameters(model_type):
    if model_type == MODEL_SMALL:
        return (768, 3072, 12, 12)
    elif model_type == MODEL_MEDIUM:
        return (1024, 4096, 24, 16)
    elif model_type == MODEL_LARGE:
        return (1280, 5120, 36, 20)
    elif model_type == MODEL_XL:
        return (1600, 6400, 48, 25)
    elif model_type == MODEL_2700M:
        return (2560, 10240, 32, 32)
    else:
        raise RuntimeError()