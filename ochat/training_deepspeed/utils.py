import mlflow
import torch.distributed as dist

def mlflow_stopper_wrapper(function):

    def mlflow_stopper(args):

        try:
            function(args)
        except:
            raise
        finally:
            RANK = dist.get_rank()
            if RANK == 0:
                mlflow.end_run()
    
    return mlflow_stopper