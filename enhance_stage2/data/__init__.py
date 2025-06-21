from importlib import import_module

from dataloader import MSDataLoader
from torch.utils.data.dataloader import default_collate

class Data:
    def __init__(self, args):
        self.loader_train = None
        self.loader_test = None
        self.loader_test_unpaired = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                # pin_memory=not args.cpu
                pin_memory=False
            )
            module_test = import_module('data.' +  args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)
            self.loader_test = MSDataLoader(
                args,
                testset,
                batch_size=1,
                shuffle=False,
                pin_memory=False
            )

        else:
            module_test = import_module('data.' +  args.data_test_unpaired.lower())
            testset_unpaired = getattr(module_test, args.data_test_unpaired)(args, train=False)
            self.loader_test_unpaired = MSDataLoader(
            args,
            testset_unpaired,
            batch_size=1,
            shuffle=False,
            pin_memory=False)