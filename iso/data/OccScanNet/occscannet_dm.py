from torch.utils.data.dataloader import DataLoader
from iso.data.OccScanNet.occscannet_dataset import OccScanNetDataset
from iso.data.OccScanNet.collate import collate_fn
import pytorch_lightning as pl
from iso.data.utils.torch_util import worker_init_fn


class OccScanNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        n_relations=4,
        batch_size=4,
        frustum_size=4,
        num_workers=6,
        interval=-1,
        train_scenes_sample=-1,
        val_scenes_sample=-1,
        v2=False,
    ):
        super().__init__()

        self.root = root
        self.n_relations = n_relations
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frustum_size = frustum_size
        
        self.train_scenes_sample = train_scenes_sample
        self.val_scenes_sample = val_scenes_sample

    def setup(self, stage=None):
        self.train_ds = OccScanNetDataset(
            split="train",
            root=self.root,
            n_relations=self.n_relations,
            fliplr=0.5,
            train_scenes_sample=self.train_scenes_sample,
            frustum_size=self.frustum_size,
            color_jitter=(0.4, 0.4, 0.4),
        )
        self.test_ds = OccScanNetDataset(
            split="val",
            root=self.root,
            n_relations=self.n_relations,
            val_scenes_sample = self.val_scenes_sample,
            frustum_size=self.frustum_size,
            fliplr=0.0,
            color_jitter=None,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )


def test():
    datamodule = OccScanNetDataModule(
        n_relations=4,
        batch_size=64,
        frustum_size=8,
    )
    datamodule.setup()
    train_data = datamodule.train_dataloader()
    print(next(iter(train_data)).shape)


if __name__ == "__main__":
    test()