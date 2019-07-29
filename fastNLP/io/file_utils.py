
import os
from pathlib import Path
from urllib.parse import urlparse
import re
import requests
import tempfile
from tqdm import tqdm
import shutil
import hashlib


PRETRAINED_BERT_MODEL_DIR = {
    'en': 'bert-large-cased-wwm.zip',
    'en-base-uncased': 'bert-base-uncased-3413b23c.zip',
    'en-base-cased': 'bert-base-cased-f89bfe08.zip',
    'en-large-uncased': 'bert-large-uncased-20939f45.zip',
    'en-large-cased': 'bert-large-cased-e0cf90fc.zip',

    'en-large-cased-wwm': 'bert-large-cased-wwm-a457f118.zip',
    'en-large-uncased-wwm': 'bert-large-uncased-wwm-92a50aeb.zip',
    'en-base-cased-mrpc': 'bert-base-cased-finetuned-mrpc-c7099855.zip',

    'cn': 'bert-base-chinese-29d0a84a.zip',
    'cn-base': 'bert-base-chinese-29d0a84a.zip',

    'multilingual': 'bert-base-multilingual-cased.zip',
    'multilingual-base-uncased': 'bert-base-multilingual-uncased.zip',
    'multilingual-base-cased': 'bert-base-multilingual-cased.zip',
}

PRETRAINED_ELMO_MODEL_DIR = {
    'en': 'elmo_en-d39843fe.tar.gz',
    'en-small': "elmo_en_Small.zip"
}

PRETRAIN_STATIC_FILES = {
    'en': 'glove.840B.300d-cc1ad5e1.tar.gz',
    'en-glove-840b-300': 'glove.840B.300d-cc1ad5e1.tar.gz',
    'en-glove-6b-50': "glove.6B.50d-a6028c70.tar.gz",
    'en-word2vec-300': "GoogleNews-vectors-negative300-be166d9d.tar.gz",
    'en-fasttext-wiki': "wiki-news-300d-1M.vec.zip",
    'cn': "tencent_cn-dab24577.tar.gz",
    'cn-fasttext': "cc.zh.300.vec-d68a9bcf.gz",
}


def cached_path(url_or_filename: str, cache_dir: Path=None) -> Path:
    """
    给定一个url或者文件名(可以是具体的文件名，也可以是文件)，先在cache_dir下寻找该文件是否存在，如果不存在则去下载, 并
        将文件放入到cache_dir中.

    :param url_or_filename: 文件的下载url或者文件路径
    :param cache_dir: 文件的缓存文件夹
    :return:
    """
    if cache_dir is None:
        dataset_cache = Path(get_default_cache_path())
    else:
        dataset_cache = cache_dir

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, dataset_cache)
    elif parsed.scheme == "" and Path(os.path.join(dataset_cache, url_or_filename)).exists():
        # File, and it exists.
        return Path(url_or_filename)
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename)
        )


def get_filepath(filepath):
    """
    如果filepath中只有一个文件，则直接返回对应的全路径.
    :param filepath:
    :return:
    """
    if os.path.isdir(filepath):
        files = os.listdir(filepath)
        if len(files)==1:
            return os.path.join(filepath, files[0])
        else:
            return filepath
    return filepath


def get_default_cache_path():
    """
    获取默认的fastNLP存放路径, 如果将FASTNLP_CACHE_PATH设置在了环境变量中，将使用环境变量的值，使得不用每个用户都去下载。

    :return:
    """
    if 'FASTNLP_CACHE_DIR' in os.environ:
        fastnlp_cache_dir = os.environ.get('FASTNLP_CACHE_DIR')
        if os.path.isdir(fastnlp_cache_dir):
            return fastnlp_cache_dir
        else:
            raise NotADirectoryError(f"{os.environ['FASTNLP_CACHE_DIR']} is not a directory.")
    fastnlp_cache_dir = os.path.expanduser(os.path.join("~", ".fastNLP"))
    return fastnlp_cache_dir


def _get_base_url(name):
    # 返回的URL结尾必须是/
    if 'FASTNLP_BASE_URL' in os.environ:
        fastnlp_base_url = os.environ['FASTNLP_BASE_URL']
        if fastnlp_base_url.endswith('/'):
            return fastnlp_base_url
        else:
            return fastnlp_base_url + '/'
    else:
        # TODO 替换
        dbbrain_url = "http://dbcloud.irocn.cn:8989/api/public/dl/"
        return dbbrain_url


def split_filename_suffix(filepath):
    """
    给定filepath返回对应的name和suffix. 如果后缀是多个点，仅支持.tar.gz类型
    :param filepath:
    :return: filename, suffix
    """
    filename = os.path.basename(filepath)
    if filename.endswith('.tar.gz'):
        return filename[:-7], '.tar.gz'
    return os.path.splitext(filename)


def get_from_cache(url: str, cache_dir: Path = None) -> Path:
    """
    尝试在cache_dir中寻找url定义的资源; 如果没有找到。则从url下载并将结果放在cache_dir下，缓存的名称由url的结果推断而来。
        如果从url中下载的资源解压后有多个文件，则返回directory的路径; 如果只有一个资源，则返回具体的路径。

    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    dir_name, suffix = split_filename_suffix(filename)

    # 寻找与它名字匹配的内容, 而不关心后缀
    match_dir_name = match_file(dir_name, cache_dir)
    if match_dir_name:
        dir_name = match_dir_name
    cache_path = cache_dir / dir_name

    # get cache path to put the file
    if cache_path.exists():
        return get_filepath(cache_path)

    # make HEAD request to check ETag TODO ETag可以用来判断资源是否已经更新了，之后需要加上
    # response = requests.head(url, headers={"User-Agent": "fastNLP"})
    # if response.status_code != 200:
    #     raise IOError(
    #         f"HEAD request failed for url {url} with status code {response.status_code}."
    #     )

    # add ETag to filename if it exists
    # etag = response.headers.get("ETag")

    if not cache_path.exists():
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        fd, temp_filename = tempfile.mkstemp()
        print("%s not found in cache, downloading to %s"%(url, temp_filename))

        # GET file object
        req = requests.get(url, stream=True, headers={"User-Agent": "fastNLP"})
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total)
        with open(temp_filename, "wb") as temp_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)
        progress.close()
        print(f"Finish download from {url}.")

        # 开始解压
        delete_temp_dir = None
        if suffix in ('.zip', '.tar.gz'):
            uncompress_temp_dir = tempfile.mkdtemp()
            delete_temp_dir = uncompress_temp_dir
            print(f"Start to uncompress file to {uncompress_temp_dir}")
            if suffix == '.zip':
                unzip_file(Path(temp_filename), Path(uncompress_temp_dir))
            else:
                untar_gz_file(Path(temp_filename), Path(uncompress_temp_dir))
            filenames = os.listdir(uncompress_temp_dir)
            if len(filenames)==1:
                if os.path.isdir(os.path.join(uncompress_temp_dir, filenames[0])):
                    uncompress_temp_dir = os.path.join(uncompress_temp_dir, filenames[0])

            cache_path.mkdir(parents=True, exist_ok=True)
            print("Finish un-compressing file.")
        else:
            uncompress_temp_dir = temp_filename
            cache_path = str(cache_path) + suffix
        success = False
        try:
            # 复制到指定的位置
            print(f"Copy file to {cache_path}")
            if os.path.isdir(uncompress_temp_dir):
                for filename in os.listdir(uncompress_temp_dir):
                    shutil.copyfile(os.path.join(uncompress_temp_dir, filename), cache_path/filename)
            else:
                shutil.copyfile(uncompress_temp_dir, cache_path)
            success = True
        except Exception as e:
            print(e)
            raise e
        finally:
            if not success:
                if cache_path.exists():
                    if cache_path.is_file():
                        os.remove(cache_path)
                    else:
                        shutil.rmtree(cache_path)
            if delete_temp_dir:
                shutil.rmtree(delete_temp_dir)
            os.close(fd)
            os.remove(temp_filename)

    return get_filepath(cache_path)


def unzip_file(file: Path, to: Path):
    # unpack and write out in CoNLL column-like format
    from zipfile import ZipFile

    with ZipFile(file, "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(to)


def untar_gz_file(file:Path, to:Path):
    import tarfile

    with tarfile.open(file, 'r:gz') as tar:
        tar.extractall(to)


def match_file(dir_name: str, cache_dir: Path) -> str:
    """
    匹配的原则是，在cache_dir下的文件: (1) 与dir_name完全一致; (2) 除了后缀以外和dir_name完全一致。
    如果找到了两个匹配的结果将报错. 如果找到了则返回匹配的文件的名称; 没有找到返回空字符串

    :param dir_name: 需要匹配的名称
    :param cache_dir: 在该目录下找匹配dir_name是否存在
    :return: str
    """
    files = os.listdir(cache_dir)
    matched_filenames = []
    for file_name in files:
        if re.match(dir_name+'$', file_name) or re.match(dir_name+'\\..*', file_name):
            matched_filenames.append(file_name)
    if len(matched_filenames)==0:
        return ''
    elif len(matched_filenames)==1:
        return matched_filenames[-1]
    else:
        raise RuntimeError(f"Duplicate matched files:{matched_filenames}, this should be caused by a bug.")

