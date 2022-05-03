def indice_collate_wrapper(func):
    """
    其功能是封装一层collate_fn,将dataset取到的tuple数据分离开，将idx打包为indices。

    :param func: 需要修饰的函数
    :return:
    """

    def wrapper(tuple_data):
        indice, ins_list = [], []
        for idx, ins in tuple_data:
            indice.append(idx)
            ins_list.append(ins)
        return indice, func(ins_list)

    return wrapper