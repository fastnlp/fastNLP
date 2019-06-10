import os

from typing import Union, Dict


def check_dataloader_paths(paths:Union[str, Dict[str, str]])->Dict[str, str]:
    """
    检查传入dataloader的文件的合法性。如果为合法路径，将返回至少包含'train'这个key的dict。类似于下面的结果
    {
        'train': '/some/path/to/', # 一定包含，建词表应该在这上面建立，剩下的其它文件应该只需要处理并index。
        'test': 'xxx' # 可能有，也可能没有
        ...
    }
    如果paths为不合法的，将直接进行raise相应的错误

    :param paths: 路径
    :return:
    """
    if isinstance(paths, str):
        if os.path.isfile(paths):
            return {'train': paths}
        elif os.path.isdir(paths):
            train_fp = os.path.join(paths, 'train.txt')
            if not os.path.isfile(train_fp):
                raise FileNotFoundError(f"train.txt is not found in folder {paths}.")
            files = {'train': train_fp}
            for filename in ['test.txt', 'dev.txt']:
                fp = os.path.join(paths, filename)
                if os.path.isfile(fp):
                    files[filename.split('.')[0]] = fp
            return files
        else:
            raise FileNotFoundError(f"{paths} is not a valid file path.")

    elif isinstance(paths, dict):
        if paths:
            if 'train' not in paths:
                raise KeyError("You have to include `train` in your dict.")
            for key, value in paths.items():
                if isinstance(key, str) and isinstance(value, str):
                    if not os.path.isfile(value):
                        raise TypeError(f"{value} is not a valid file.")
                else:
                    raise TypeError("All keys and values in paths should be str.")
            return paths
        else:
            raise ValueError("Empty paths is not allowed.")
    else:
        raise TypeError(f"paths only supports str and dict. not {type(paths)}.")


