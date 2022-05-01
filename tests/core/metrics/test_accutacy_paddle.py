import os

import pytest
import paddle
import paddle.distributed
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
from fastNLP.core.metrics import Accuracy
from fastNLP.core.drivers.paddle_driver.fleet_launcher import FleetLauncher


############################################################################
#
# 测试 单机单卡情况下的Accuracy
#
############################################################################
@pytest.mark.paddle
def test_accuracy_single():
    pred = paddle.to_tensor([[1.19812393, -0.82041764, -0.53517765, -0.73061031, -1.45006669,
                              0.46514302],
                             [-0.85775983, -2.18273783, -1.07505429, -1.45561373, 0.40011844,
                              1.02202022],
                             [-0.39487389, 0.65682763, -0.62424040, 0.53692561, -0.28390560,
                              -0.02559055],
                             [-0.22586937, -0.07676325, -0.95977223, 0.36395910, -0.91758579,
                              -0.83857095],
                             [0.25136873, 2.49652624, 1.06251311, 1.60194016, 1.01451588,
                              0.08403367],
                             [0.10844281, 1.19017303, -0.11378096, 1.12686944, -0.08654942,
                              0.48605862],
                             [1.27320433, -1.13902378, 1.47072780, -0.98665696, -0.42589864,
                              0.64618838],
                             [0.83809763, -0.05356205, 0.03042423, -0.28371972, 0.81611472,
                              -0.45802942],
                             [0.38535264, 0.09721313, 2.27187467, 0.32045507, -0.20711982,
                              -0.13550705],
                             [-0.75228405, -1.34161997, 1.08697927, 0.33218071, -1.19470012,
                              2.58735061]])
    tg = paddle.to_tensor([1, 2, 1, 3, 5, 4, 4, 2, 1, 5])
    acc_metric = Accuracy()
    acc_metric.update(pred, tg)
    result = acc_metric.get_metric()
    true_result = {'acc': 0.3}
    assert true_result == result


############################################################################
#
# 测试 单机多卡情况下的Accuracy
#
############################################################################
def test_accuracy_ddp():
    launcher = FleetLauncher(devices=[0, 1])
    launcher.launch()
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    if fleet.is_server():
        pass
    elif fleet.is_worker():
        print(os.getenv("PADDLE_TRAINER_ID"))
