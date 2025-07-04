import time
import logging
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Проверка наличия GPU
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
        batch_norm: bool = False
    ):
        """
        Многослойный перцептрон с опциональным Dropout и BatchNorm.

        Args:
            input_dim: размер входа
            output_dim: количество классов
            hidden_layers: список с количеством нейронов в скрытых слоях
            dropout: вероятность dropout (0 - без dropout)
            batch_norm: использовать BatchNorm или нет
        """
        super().__init__()
        layers = []
        prev_dim = input_dim

        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
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
) -> float:
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
    device
) -> Dict[str, List[float]]:
    """
    Обучает модель и возвращает словарь с метриками по эпохам:
    train_loss, train_acc, val_loss, val_acc
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        logger.info(
            f"Epoch {epoch:02d}: "
            f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

    return history


def plot_learning_curves(
    history: Dict[str, List[float]],
    depth: int,
    dropout: float,
    batch_norm: bool,
    save_path: Optional[str] = None
):
    """
    Визуализирует кривые train/test accuracy и loss.
    """
    epochs = np.arange(1, len(history['train_loss']) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(epochs, history['train_loss'], label='Train Loss')
    axs[0].plot(epochs, history['val_loss'], label='Test Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(f'Loss Curves (Depth={depth}, Dropout={dropout}, BatchNorm={batch_norm})')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(epochs, history['train_acc'], label='Train Accuracy')
    axs[1].plot(epochs, history['val_acc'], label='Test Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title(f'Accuracy Curves (Depth={depth}, Dropout={dropout}, BatchNorm={batch_norm})')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved learning curves to {save_path}")
    plt.show()


def experiment_depth_overfitting(
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    depths: List[int],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.0,
    batch_norm: bool = False
):
    """
    Эксперимент с разной глубиной сети, анализ переобучения, визуализация.

    Args:
        train_ds, test_ds: датасеты
        depths: список глубин (число слоев включая выходный)
        epochs: число эпох обучения
        batch_size: размер батча
        lr: learning rate
        dropout: dropout rate
        batch_norm: использовать batch norm
    """
    input_dim = train_ds[0][0].shape[0]
    output_dim = len(set(y for _, y in train_ds))

    # Для валидации выделим 20% из train_ds
    val_len = int(len(train_ds) * 0.2)
    train_len = len(train_ds) - val_len
    train_subds, val_ds = random_split(train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = []

    for depth in depths:
        # depth = общее число слоев, значит hidden = depth-1 для линейного классификатора hidden=0
        hidden_layers = []
        if depth > 1:
            # равномерно по 100 нейронов в каждом скрытом слое
            hidden_layers = [100] * (depth - 1)

        model = MLP(input_dim, output_dim, hidden_layers, dropout, batch_norm).to(device)
        logger.info(f"Training model with depth={depth}, dropout={dropout}, batch_norm={batch_norm}")

        start_time = time.time()
        history = train_model(model, train_loader, val_loader, epochs, lr, device)
        elapsed = time.time() - start_time
        logger.info(f"Training time: {elapsed:.2f} sec")

        # Финальная оценка на тесте
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        logger.info(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

        results.append({
            'depth': depth,
            'history': history,
            'time': elapsed,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'dropout': dropout,
            'batch_norm': batch_norm
        })

        plot_learning_curves(history, depth, dropout, batch_norm)

    # Анализ результатов: построим графики финальной точности и времени
    plot_summary(results)


def plot_summary(results: List[dict]):
    """
    Построение сравнительного графика accuracy и времени обучения по глубине и конфигурации.
    """
    import seaborn as sns
    sns.set(style="whitegrid")

    # Группируем по dropout и batch_norm
    configs = {}
    for r in results:
        key = (r['dropout'], r['batch_norm'])
        if key not in configs:
            configs[key] = []
        configs[key].append(r)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    for (dropout, batch_norm), res_list in configs.items():
        depths = [r['depth'] for r in res_list]
        test_accs = [r['test_acc'] for r in res_list]
        times = [r['time'] for r in res_list]
        label = f"Dropout={dropout}, BatchNorm={batch_norm}"

        axs[0].plot(depths, test_accs, marker='o', label=label)
        axs[1].plot(depths, times, marker='o', label=label)

    axs[0].set_xlabel('Depth (total layers)')
    axs[0].set_ylabel('Test Accuracy')
    axs[0].set_title('Test Accuracy vs Depth')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_xlabel('Depth (total layers)')
    axs[1].set_ylabel('Training Time (seconds)')
    axs[1].set_title('Training Time vs Depth')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Логируем краткий итог
    for r in results:
        logger.info(
            f"Depth={r['depth']}, Dropout={r['dropout']}, BatchNorm={r['batch_norm']}, "
            f"Test Acc={r['test_acc']:.4f}, Time={r['time']:.2f}s"
        )


def test_simple_run():
    """
    Тестирование на простой модели без dropout и batch norm, 1 слой.
    Проверяем, что обучение проходит и метрики в разумных пределах.
    """
    logger.info("Running simple test...")
    train_ds, test_ds = load_and_prepare_data(test_size=0.5, random_state=0)

    depths = [1]
    experiment_depth_overfitting(
        train_ds, test_ds,
        depths=depths,
        epochs=5,
        batch_size=32,
        lr=1e-3,
        dropout=0.0,
        batch_norm=False
    )
    logger.info("Simple test passed.")


if __name__ == "__main__":
    # Запускаем тест
    test_simple_run()

    # Параметры эксперимента
    train_ds, test_ds = load_and_prepare_data()
    depths = [1, 2, 3, 5, 7]

    # Эксперимент без регуляризации
    logger.info("=== Experiment: No Dropout, No BatchNorm ===")
    experiment_depth_overfitting(
        train_ds, test_ds,
        depths=depths,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        dropout=0.0,
        batch_norm=False
    )

    # Эксперимент с Dropout
    logger.info("=== Experiment: Dropout=0.3, No BatchNorm ===")
    experiment_depth_overfitting(
        train_ds, test_ds,
        depths=depths,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        dropout=0.3,
        batch_norm=False
    )

    # Эксперимент с BatchNorm
    logger.info("=== Experiment: No Dropout, BatchNorm=True ===")
    experiment_depth_overfitting(
        train_ds, test_ds,
        depths=depths,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        dropout=0.0,
        batch_norm=True
    )

    # Эксперимент с Dropout и BatchNorm
    logger.info("=== Experiment: Dropout=0.3, BatchNorm=True ===")
    experiment_depth_overfitting(
        train_ds, test_ds,
        depths=depths,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        dropout=0.3,
        batch_norm=True
    )
