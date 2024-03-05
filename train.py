import torch
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from model import FGCM_Model

torch.set_float32_matmul_precision('medium') 

def train_model(cfg, model, data_module, logger):  
    # Callbacks      
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        monitor='val_acc',
        filename='{cfg.backbone}_{epoch:02d}_{acc:.2f}',
        save_top_k=1,
        mode='max',
        verbose=True,
    )
    LR_monitor_callback = LearningRateMonitor(
        logging_interval='epoch', 
    )
    Rich_pbar_callback = RichProgressBar()
     
    # Initialize trainer
    trainer = Trainer(
        max_epochs=cfg.epochs,
        log_every_n_steps=1,
        callbacks = [
            LR_monitor_callback, 
            checkpoint_callback,
            Rich_pbar_callback
        ],
        logger=logger,
        accelerator='gpu',
        devices=1,
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    best_model_path = checkpoint_callback.best_model_path 
       
    return trainer, best_model_path

def test_model(best_model_path, data_module, logger):
    print(f"Loading best model from {best_model_path}")

    # Initialize trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        logger=logger,
    )
    
    # Load the best model
    best_model = FGCM_Model.load_from_checkpoint(best_model_path)
    
    # Run the test using the best model
    trainer.test(best_model, datamodule=data_module)
