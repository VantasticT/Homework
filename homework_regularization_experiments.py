import time
import logging
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def load_and_prepare_data(test_size=0.3, random_state=42) -> Tuple[TensorDataset, TensorDataset]:
    """
    Загружает digits, масштабирует и возвращает TensorDataset для train и test.
    """
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    logger.info(f"Data prepared: train size={len(train_ds)}, test size={len(test_ds)}")
    return train_ds, test_ds


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        dropout: float = 0.0,
        batch_norm: bool = False,
        batch_norm_momentum: float = 0.1,
        dropout_schedule: Optional[List[float]] = None,
        adaptive_dropout: bool = False,
        adaptive_bn: bool = False,
        adaptive_bn_momentum_schedule: Optional[List[float]] = None,
    ):
        """
        Многослойный перцептрон с опциональным Dropout и BatchNorm.
        Поддержка адаптивного Dropout и BatchNorm (изменение коэффициентов по слоям).

        Args:
            input_dim: размер входа
            output_dim: количество классов
            hidden_layers: список с количеством нейронов в скрытых слоях
            dropout: фиксированный dropout коэффициент (если adaptive_dropout==False)
            batch_norm: использовать BatchNorm или нет
            batch_norm_momentum: фиксированный momentum для BatchNorm
            dropout_schedule: список dropout для каждого слоя (если adaptive_dropout==True)
            adaptive_dropout: использовать ли dropout_schedule
            adaptive_bn: использовать ли adaptive_bn_momentum_schedule
            adaptive_bn_momentum_schedule: список momentum для BatchNorm по слоям
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        n_layers = len(hidden_layers)

        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, h))
            # BatchNorm с адаптивным momentum
            if batch_norm:
                mom = batch_norm_momentum
                if adaptive_bn and adaptive_bn_momentum_schedule is not None:
                    mom = adaptive_bn_momentum_schedule[i]
                layers.append(nn.BatchNorm1d(h, momentum=mom))
            layers.append(nn.ReLU(inplace=True))
            # Dropout с адаптивным коэффициентом
            if dropout > 0 or (adaptive_dropout and dropout_schedule is not None):
                d = dropout
                if adaptive_dropout and dropout_schedule is not None:
                    d = dropout_schedule[i]
                if d > 0:
                    layers.append(nn.Dropout(d))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device
) -> Dict[str, List[float]]:
    """
    Обучает модель и возвращает словарь с метриками по эпохам:
    train_loss, train_acc, val_loss, val_acc
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logger.debug(
            f"Epoch {epoch:02d}: "
            f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

    return history


def plot_training_history(histories: Dict[str, Dict[str, List[float]]]):
    """
    Визуализация истории обучения для разных техник регуляризации.
    """
    plt.figure(figsize=(14, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    for name, hist in histories.items():
        plt.plot(hist['train_loss'], label=f"{name} train")
        plt.plot(hist['val_loss'], linestyle='--', label=f"{name} val")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    for name, hist in histories.items():
        plt.plot(hist['train_acc'], label=f"{name} train")
        plt.plot(hist['val_acc'], linestyle='--', label=f"{name} val")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_weight_distributions(
    models: Dict[str, nn.Module],
    layer_indices: Optional[List[int]] = None
):
    """
    Визуализация распределения весов для указанных слоев моделей.

    Args:
        models: словарь {name: model}
        layer_indices: индексы слоев (Linear) для анализа (по умолчанию первые 3)
    """
    if layer_indices is None:
        layer_indices = [0, 3, 6]  # обычно Linear находятся на позициях 0,3,6

    plt.figure(figsize=(14, 4 * len(layer_indices)))

    for i, layer_idx in enumerate(layer_indices):
        plt.subplot(len(layer_indices), 1, i + 1)
        for name, model in models.items():
            # Получаем слой Linear
            try:
                layer = model.net[layer_idx]
                if not isinstance(layer, nn.Linear):
                    # Поиск ближайшего Linear перед layer_idx
                    # (на всякий случай)
                    layer = next(l for l in model.net if isinstance(l, nn.Linear))
            except Exception:
                layer = next(l for l in model.net if isinstance(l, nn.Linear))

            weights = layer.weight.detach().cpu().numpy().flatten()
            sns.kdeplot(weights, label=name, fill=True, alpha=0.4)

        plt.title(f"Weight distribution for layer index {layer_idx} (Linear layer)")
        plt.xlabel("Weight value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def experiment_regularization_comparison(
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_layers: List[int] = [256, 128, 64],
):
    """
    Эксперимент 3.1: сравнение техник регуляризации.

    Техники:
        - no_reg: без регуляризации
        - dropout: с Dropout (0.1, 0.3, 0.5)
        - batchnorm: только BatchNorm
        - dropout+batchnorm
        - L2 (weight_decay)
    """
    input_dim = train_ds[0][0].shape[0]
    output_dim = len(set(y for _, y in train_ds))

    val_len = int(len(train_ds) * 0.2)
    train_len = len(train_ds) - val_len
    train_subds, val_ds = random_split(train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    histories = {}
    models = {}

    # 1. No regularization
    logger.info("Training no regularization model")
    model_no_reg = MLP(input_dim, output_dim, hidden_layers, dropout=0.0, batch_norm=False).to(device)
    history_no_reg = train_model(model_no_reg, train_loader, val_loader, epochs, lr, weight_decay=0.0, device=device)
    histories['no_reg'] = history_no_reg
    models['no_reg'] = model_no_reg

    # 2. Dropout only (several rates)
    for d_rate in [0.1, 0.3, 0.5]:
        name = f"dropout_{d_rate}"
        logger.info(f"Training model with Dropout {d_rate}")
        model = MLP(input_dim, output_dim, hidden_layers, dropout=d_rate, batch_norm=False).to(device)
        history = train_model(model, train_loader, val_loader, epochs, lr, weight_decay=0.0, device=device)
        histories[name] = history
        models[name] = model

    # 3. BatchNorm only
    logger.info("Training model with BatchNorm only")
    model_bn = MLP(input_dim, output_dim, hidden_layers, dropout=0.0, batch_norm=True).to(device)
    history_bn = train_model(model_bn, train_loader, val_loader, epochs, lr, weight_decay=0.0, device=device)
    histories['batchnorm'] = history_bn
    models['batchnorm'] = model_bn

    # 4. Dropout + BatchNorm (dropout=0.3)
    logger.info("Training model with Dropout 0.3 + BatchNorm")
    model_dn_bn = MLP(input_dim, output_dim, hidden_layers, dropout=0.3, batch_norm=True).to(device)
    history_dn_bn = train_model(model_dn_bn, train_loader, val_loader, epochs, lr, weight_decay=0.0, device=device)
    histories['dropout_0.3_batchnorm'] = history_dn_bn
    models['dropout_0.3_batchnorm'] = model_dn_bn

    # 5. L2 regularization (weight_decay)
    logger.info("Training model with L2 regularization (weight_decay=1e-4)")
    model_l2 = MLP(input_dim, output_dim, hidden_layers, dropout=0.0, batch_norm=False).to(device)
    history_l2 = train_model(model_l2, train_loader, val_loader, epochs, lr, weight_decay=1e-4, device=device)
    histories['l2_reg'] = history_l2
    models['l2_reg'] = model_l2

    # Визуализация обучения
    plot_training_history(histories)
    # Визуализация распределения весов
    plot_weight_distributions(models)

    # Итоговые точности на тесте
    criterion = nn.CrossEntropyLoss()
    logger.info("Final test accuracies:")
    for name, model in models.items():
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        logger.info(f"Model '{name}': Test accuracy = {test_acc:.4f}")


def experiment_adaptive_regularization(
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_layers: List[int] = [256, 128, 64],
):
    """
    Эксперимент 3.2: адаптивные техники регуляризации:
        - Dropout с меняющимся коэффициентом по слоям
        - BatchNorm с разным momentum по слоям
        - Комбинирование нескольких техник
        - Анализ влияния на разные слои сети
    """
    input_dim = train_ds[0][0].shape[0]
    output_dim = len(set(y for _, y in train_ds))

    val_len = int(len(train_ds) * 0.2)
    train_len = len(train_ds) - val_len
    train_subds, val_ds = random_split(train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    histories = {}
    models = {}

    # 1. Adaptive Dropout: увеличивается с глубиной
    dropout_schedule_inc = [0.1, 0.3, 0.5]
    logger.info(f"Training adaptive dropout increasing schedule: {dropout_schedule_inc}")
    model_adapt_dropout_inc = MLP(
        input_dim,
        output_dim,
        hidden_layers,
        dropout=0.0,
        batch_norm=False,
        dropout_schedule=dropout_schedule_inc,
        adaptive_dropout=True
    ).to(device)
    hist_adapt_dropout_inc = train_model(model_adapt_dropout_inc, train_loader, val_loader, epochs, lr, 0.0, device)
    histories['adaptive_dropout_inc'] = hist_adapt_dropout_inc
    models['adaptive_dropout_inc'] = model_adapt_dropout_inc

    # 2. Adaptive Dropout: уменьшается с глубиной
    dropout_schedule_dec = [0.5, 0.3, 0.1]
    logger.info(f"Training adaptive dropout decreasing schedule: {dropout_schedule_dec}")
    model_adapt_dropout_dec = MLP(
        input_dim,
        output_dim,
        hidden_layers,
        dropout=0.0,
        batch_norm=False,
        dropout_schedule=dropout_schedule_dec,
        adaptive_dropout=True
    ).to(device)
    hist_adapt_dropout_dec = train_model(model_adapt_dropout_dec, train_loader, val_loader, epochs, lr, 0.0, device)
    histories['adaptive_dropout_dec'] = hist_adapt_dropout_dec
    models['adaptive_dropout_dec'] = model_adapt_dropout_dec

    # 3. Adaptive BatchNorm momentum: уменьшается с глубиной
    bn_momentum_schedule = [0.3, 0.1, 0.05]
    logger.info(f"Training adaptive BatchNorm momentum schedule: {bn_momentum_schedule}")
    model_adapt_bn = MLP(
        input_dim,
        output_dim,
        hidden_layers,
        dropout=0.0,
        batch_norm=True,
        adaptive_bn=True,
        adaptive_bn_momentum_schedule=bn_momentum_schedule
    ).to(device)
    hist_adapt_bn = train_model(model_adapt_bn, train_loader, val_loader, epochs, lr, 0.0, device)
    histories['adaptive_batchnorm_momentum'] = hist_adapt_bn
    models['adaptive_batchnorm_momentum'] = model_adapt_bn

    # 4. Комбинация adaptive Dropout (inc) + adaptive BatchNorm momentum
    logger.info("Training combined adaptive Dropout + adaptive BatchNorm")
    model_combined = MLP(
        input_dim,
        output_dim,
        hidden_layers,
        dropout=0.0,
        batch_norm=True,
        adaptive_dropout=True,
        dropout_schedule=dropout_schedule_inc,
        adaptive_bn=True,
        adaptive_bn_momentum_schedule=bn_momentum_schedule
    ).to(device)
    hist_combined = train_model(model_combined, train_loader, val_loader, epochs, lr, 0.0, device)
    histories['combined_adaptive'] = hist_combined
    models['combined_adaptive'] = model_combined

    # Визуализация
    plot_training_history(histories)
    plot_weight_distributions(models)

    # Итоговые точности на тесте
    criterion = nn.CrossEntropyLoss()
    logger.info("Final test accuracies for adaptive regularization:")
    for name, model in models.items():
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        logger.info(f"Model '{name}': Test accuracy = {test_acc:.4f}")


def test_simple_run():
    """
    Быстрый тест для проверки корректности экспериментов.
    """
    logger.info("Running simple test for regularization experiments...")
    train_ds, test_ds = load_and_prepare_data(test_size=0.5, random_state=0)

    # Минимальное число эпох для теста
    epochs = 3
    batch_size = 32
    lr = 1e-3
    hidden_layers = [64, 32, 16]

    experiment_regularization_comparison(
        train_ds, test_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_layers=hidden_layers
    )

    experiment_adaptive_regularization(
        train_ds, test_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_layers=hidden_layers
    )
    logger.info("Simple test passed.")


if __name__ == "__main__":
    # Тестирование
    test_simple_run()

    # Основные параметры
    train_ds, test_ds = load_and_prepare_data()
    epochs = 40
    batch_size = 64
    lr = 1e-3
    hidden_layers = [256, 128, 64]

    logger.info("=== Experiment 3.1: Regularization Techniques Comparison ===")
    experiment_regularization_comparison(
        train_ds, test_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_layers=hidden_layers
    )

    logger.info("=== Experiment 3.2: Adaptive Regularization Techniques ===")
    experiment_adaptive_regularization(
        train_ds, test_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_layers=hidden_layers
    )
