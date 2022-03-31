import kfp
import kfp.components as comp
from kfp import dsl
from kfp import onprem

@dsl.pipeline(
    name='kube-srgan',
    description='kube-srgan-test'
)

def pipelinemaker():

    data   = dsl.ContainerOp(
            name = "load, preprocess data pipeline",
            image = "parks602/srgan_kube_data:0.1",
            ).set_display_name('preprocess data')\
                .apply(onprem.mount_pvc('data-pvc', volume_name='data', volume_mount_path='/data'))

    train   = dsl.ContainerOp(
            name = "train srgan model",
            image = "parks602/srgan_kube_train:0.1",
            ).set_display_name('train')\
                .apply(onprem.mount_pvc('data-pvc', volume_name='data', volume_mount_path='/data'))\
                .apply(onprem.mount_pvc('train-model-pvc', volume_name='train-model', volume_mount_path='/model'))

    test    = dsl.ContainerOp(
            name = "test srgan model",
            image = "parks602/srgan_kube_test:0.1",
            ).set_display_name('test')\
                .apply(onprem.mount_pvc('data-pvc', volume_name='data', volume_mount_path='data'))\
                .apply(onprem.mount_pvc('train-model-pvc', volume_name='train-model', volume_mount_path='/model'))
if __name__ == "__main__":
    pipelinemaker
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipelinemaker, __file__ + ".tar.gz")
