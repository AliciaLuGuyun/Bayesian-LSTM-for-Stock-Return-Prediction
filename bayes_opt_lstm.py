# bayes_opt_lstm.py
import os
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from model import BayesianLSTM
from train import LSTMTrainer
from predict_mc import predict_with_uncertainty
from dataset import create_lstm_dataloaders
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime

class BayesianLSTMOptimizer:
    def __init__(self, config):
        self.config = config
        self.device = self._setup_environment()
        self.output_dir = self._prepare_output_directories()
        self.log_train, self.log_test = self._load_data()
        
    def _setup_environment(self):
        """Initialize all random seeds and device"""
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _prepare_output_directories(self):
        """Create necessary output directories"""
        base_dir = Path(__file__).resolve().parent
        output_dir = base_dir / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plots").mkdir(exist_ok=True)
        (output_dir / "models").mkdir(exist_ok=True)
        (output_dir / "predictions").mkdir(exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        return output_dir
    
    def _load_data(self):
        """Load and preprocess the stock data"""
        base_dir = Path(__file__).resolve().parent
        train_df = pd.read_csv(base_dir / "stock_close_prices_train.csv", parse_dates=["Date"])
        test_df = pd.read_csv(base_dir / "stock_close_prices_test.csv", parse_dates=["Date"])
        
        def compute_log_returns(df):
            return np.log(df / df.shift(1)).dropna()
        
        return compute_log_returns(train_df.set_index("Date")), compute_log_returns(test_df.set_index("Date"))
    
    def _calculate_metrics(self, true_values, mean_preds, std_preds):
        """Calculate all evaluation metrics"""
        mse = np.mean((true_values - mean_preds) ** 2)
        z = 1.96
        lower = mean_preds - z * std_preds
        upper = mean_preds + z * std_preds
        picp = np.mean((true_values >= lower) & (true_values <= upper))
        mpiw = np.mean(upper - lower)
        score = -mse + 0.001 * picp - 0.1 * mpiw
        return {
            "mse": mse,
            "picp": picp,
            "mpiw": mpiw,
            "score": score,
            "lower_ci": lower,
            "upper_ci": upper
        }
    
    def _create_objective_function(self, stock):
        """Create the objective function for Bayesian optimization"""
        train_series = self.log_train[stock].dropna()
        test_series = self.log_test[stock].dropna()
        
        # Skip if not enough test data
        if len(test_series) < self.config.WINDOW_SIZE + 5:
            print(f"Skipping {stock} - insufficient test data")
            return None
        
        train_loader, test_seq, test_target = create_lstm_dataloaders(
            train_series.values, 
            test_series.values, 
            window_size=self.config.WINDOW_SIZE, 
            batch_size=self.config.BATCH_SIZE
        )
        
        def objective(hidden_dim, dropout, lr):
            hidden_dim = int(hidden_dim)
            model = BayesianLSTM(
                input_dim=1,
                hidden_dim=hidden_dim,
                dropout=dropout
            ).to(self.device)
            
            # Train the model
            trainer = LSTMTrainer(model, device=self.device, lr=lr)
            trainer.train(train_loader, epochs=self.config.EPOCHS)
            
            # Evaluate the model
            mean_preds, std_preds = predict_with_uncertainty(
                model, test_seq, device=self.device, mc_samples=self.config.MC_SAMPLES
            )
            true_values = test_target.squeeze().numpy()
            
            metrics = self._calculate_metrics(true_values, mean_preds, std_preds)
            return metrics["score"]
        
        return objective, train_loader, test_seq, test_target, test_series
    
    def _save_results(self, stock, model, params, metrics, test_series, test_target, predictions):
        """Save all results for a single stock"""
        # Save model
        model_path = self.output_dir / "models" / f"{stock}_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Save parameters
        params_path = self.output_dir / "models" / f"{stock}_params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f)
        
        # Save predictions
        pred_df = pd.DataFrame({
            "Date": test_series.index[self.config.WINDOW_SIZE:],
            "True": test_target.squeeze().numpy(),
            "Predicted_Mean": predictions["mean_preds"],
            "Lower_95CI": predictions["lower_ci"],
            "Upper_95CI": predictions["upper_ci"]
        })
        pred_df.to_csv(self.output_dir / "predictions" / f"{stock}_predictions.csv", index=False)
        
        # Plot results
        self._plot_results(stock, test_series, test_target, predictions)
        
        return {
            "stock": stock,
            "params": params,
            "metrics": metrics,
            "model_path": str(model_path),
            "params_path": str(params_path)
        }
    
    def _plot_results(self, stock, test_series, test_target, predictions):
        """Generate and save prediction plots"""
        plt.figure(figsize=(12, 6))
        dates = test_series.index[self.config.WINDOW_SIZE:]
        
        plt.plot(dates, test_target.squeeze().numpy(), label="True Return", color="black")
        plt.plot(dates, predictions["mean_preds"], label="Predicted Mean", color="blue")
        plt.fill_between(
            dates, 
            predictions["lower_ci"], 
            predictions["upper_ci"], 
            color="blue", 
            alpha=0.2, 
            label="95% Confidence Interval"
        )
        
        plt.title(f"Prediction with Uncertainty: {stock}")
        plt.xlabel("Date")
        plt.ylabel("Log Return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / f"{stock}_prediction.png")
        plt.close()
    
    def optimize_and_evaluate(self):
        """Main optimization and evaluation pipeline"""
        results = {}
        
        for stock in self.log_train.columns:
            print(f"\n=== Processing {stock} ===")
            
            # Create objective function
            objective_func, train_loader, test_seq, test_target, test_series = self._create_objective_function(stock)
            if objective_func is None:
                continue
            
            # Setup Bayesian optimization
            optimizer = BayesianOptimization(
                f=objective_func,
                pbounds={
                    'hidden_dim': (32, 128),
                    'dropout': (0.1, 0.5),
                    'lr': (1e-4, 1e-2)
                },
                random_state=self.config.SEED,
                verbose=2
            )
            
            # Add logger
            logger = JSONLogger(path=str(self.output_dir / "logs" / f"{stock}_optimization_logs.json"))
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            
            # Run optimization
            optimizer.maximize(
                init_points=self.config.INIT_POINTS,
                n_iter=self.config.N_ITER
            )
            
            # Get best parameters
            best_params = optimizer.max['params']
            best_params['hidden_dim'] = int(best_params['hidden_dim'])
            
            # Train final model with best parameters
            final_model = BayesianLSTM(
                input_dim=1,
                hidden_dim=best_params['hidden_dim'],
                dropout=best_params['dropout']
            ).to(self.device)
            
            trainer = LSTMTrainer(final_model, device=self.device, lr=best_params['lr'])
            trainer.train(train_loader, epochs=self.config.EPOCHS)
            
            # Final evaluation
            mean_preds, std_preds = predict_with_uncertainty(
                final_model, test_seq, device=self.device, mc_samples=self.config.MC_SAMPLES
            )
            metrics = self._calculate_metrics(test_target.squeeze().numpy(), mean_preds, std_preds)
            
            # Save results
            results[stock] = self._save_results(
                stock=stock,
                model=final_model,
                params=best_params,
                metrics=metrics,
                test_series=test_series,
                test_target=test_target,
                predictions={
                    "mean_preds": mean_preds,
                    "lower_ci": metrics["lower_ci"],
                    "upper_ci": metrics["upper_ci"]
                }
            )
        
        # Save summary
        self._save_summary(results)
        return results
    
    def _save_summary(self, results):
        """Save summary of all stocks' results"""
        summary_path = self.output_dir / "summary.csv"
        
        summary_data = []
        for stock, res in results.items():
            summary_data.append({
                "stock": stock,
                "hidden_dim": res["params"]["hidden_dim"],
                "dropout": res["params"]["dropout"],
                "lr": res["params"]["lr"],
                "mse": res["metrics"]["mse"],
                "picp": res["metrics"]["picp"],
                "mpiw": res["metrics"]["mpiw"],
                "score": res["metrics"]["score"],
                "model_path": res["model_path"],
                "params_path": res["params_path"]
            })
        
        pd.DataFrame(summary_data).to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

class Config:
    SEED = 42
    WINDOW_SIZE = 20
    BATCH_SIZE = 32
    EPOCHS = 20
    MC_SAMPLES = 100
    INIT_POINTS = 5
    N_ITER = 10

if __name__ == "__main__":
    optimizer = BayesianLSTMOptimizer(Config())
    results = optimizer.optimize_and_evaluate()