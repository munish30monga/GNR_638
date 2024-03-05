import wandb
import argparse
import yaml
import munch
from train import train_model, test_model
from dataloader import CUB_DataModule, dataset_summary
from model import FGCM_Model
from lightning.pytorch.loggers import WandbLogger

project_name = 'run'
# wandb.init(mode='dryrun')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fine Grained Classification Model.")
    parser.add_argument(
        "--config_file",
        default="configs/aio.yaml",
        type=str,
        help="path to config file.",
    )
    args = parser.parse_args()

    # Load config yaml file as nested object
    cfg = yaml.safe_load(open(args.config_file, "r"))
    cfg = munch.munchify(cfg)
    
    wandb.init(name=f'{cfg.backbone}_{cfg.loss_function}_fpn:{cfg.use_fpn}_aug:{cfg.use_augm}_ls:{cfg.label_smoothing}_temp:{cfg.temperature}_gamma:{cfg.gamma}_scheduler:{cfg.scheduler}_{project_name}')
    wandb.config.update(cfg)
    wandb.save(f'./configs/aio.yaml')

    print(f"=> Configurations:\n{cfg}")
    
    logger = WandbLogger(project='GNR_638', log_model='all', name=f'{project_name}')
    dataset_summary_dict = dataset_summary(cfg)
    data_module = CUB_DataModule(cfg)
    data_module.setup()
    num_classes = dataset_summary_dict['num_classes']

    model = FGCM_Model(cfg, num_classes)
    print(f"=> Fine-Grained Classification Model is build using '{cfg.backbone}' as base model.")
    
    # Training Model
    trainer, best_model_path = train_model(cfg, model, data_module, logger)
    
    # Testing Model
    test_model(best_model_path, data_module, logger)
    
    wandb.finish()
    
    