import torch
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from model import FGCM_Model

torch.set_float32_matmul_precision('medium') 

def train_model(cfg , model, data_module, max_epochs, accelerator, devices, logger):  
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
    Rich_pbar_callback = RichProgressBar(leave = True)
     
    # Initialize trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        callbacks = [
            LR_monitor_callback, 
            checkpoint_callback,
            Rich_pbar_callback
        ],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    print(f"=> Testing the best model on test set.")
    # Load the best model saved by the checkpoint callback
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    if best_model_path:
        best_model = FGCM_Model.load_from_checkpoint(best_model_path)
    # Run the test using the best model
    trainer.test(best_model, datamodule=data_module)
    
    return trainer, best_model