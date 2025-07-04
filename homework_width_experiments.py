# homework_width_experiments.py

import time
import logging
from typing import List, Tuple, Dict, Optional

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


def count_parameters(model: nn.Module) -> int:
    """
    Возвращает количество обучаемых параметров модели.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def experiment_width_comparison(
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    layer_widths: Dict[str, List[int]],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.0,
    batch_norm: bool = False
):
    """
    Эксперимент: сравнение моделей с разной шириной при фиксированной глубине.

    Args:
        train_ds, test_ds: датасеты
        layer_widths: словарь с именами конфигураций и списками ширин слоев (3 слоя)
        epochs, batch_size, lr, dropout, batch_norm: параметры обучения
    """
    input_dim = train_ds[0][0].shape[0]
    output_dim = len(set(y for _, y in train_ds))

    # Разбиваем train на train/val для контроля
    val_len = int(len(train_ds) * 0.2)
    train_len = len(train_ds) - val_len
    train_subds, val_ds = random_split(train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = []

    for name, widths in layer_widths.items():
        logger.info(f"Training model with width config '{name}': layers={widths}")
        model = MLP(input_dim, output_dim, widths, dropout, batch_norm).to(device)

        start_time = time.time()
        history = train_model(model, train_loader, val_loader, epochs, lr, device)
        elapsed = time.time() - start_time

        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)

        n_params = count_parameters(model)

        logger.info(f"Config '{name}': Test acc={test_acc:.4f}, Test loss={test_loss:.4f}, Params={n_params}, Time={elapsed:.2f}s")

        results.append({
            'name': name,
            'widths': widths,
            'history': history,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'params': n_params,
            'time': elapsed
        })

    plot_width_comparison(results)


def plot_width_comparison(results: List[dict]):
    """
    Визуализирует сравнение точности, времени и количества параметров для разных конфигураций ширины.
    """
    names = [r['name'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    times = [r['time'] for r in results]
    params = [r['params'] for r in results]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(x=names, y=test_accs, ax=axs[0])
    axs[0].set_title('Test Accuracy')
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Width Config')

    sns.barplot(x=names, y=times, ax=axs[1])
    axs[1].set_title('Training Time (sec)')
    axs[1].set_ylabel('Time [s]')
    axs[1].set_xlabel('Width Config')

    sns.barplot(x=names, y=params, ax=axs[2])
    axs[2].set_title('Number of Parameters')
    axs[2].set_ylabel('Params')
    axs[2].set_xlabel('Width Config')
    axs[2].ticklabel_format(axis='y', style='sci', scilimits=(3, 4))

    plt.suptitle('Comparison of Different Width Configurations')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def generate_width_patterns(
    base_width: int,
    depth: int,
    pattern: str
) -> List[int]:
    """
    Генерирует список ширин слоев по паттерну.

    Args:
        base_width: базовое число нейронов
        depth: число слоев
        pattern: 'constant', 'widening', 'narrowing'

    Returns:
        Список ширин длины depth
    """
    if depth < 1:
        raise ValueError("Depth must be >=1")

    if pattern == 'constant':
        return [base_width] * depth
    elif pattern == 'widening':
        # Каждый следующий слой шире в 2 раза
        return [base_width * (2 ** i) for i in range(depth)]
    elif pattern == 'narrowing':
        # Каждый следующий слой уже в 2 раза
        return [base_width // (2 ** i) if base_width // (2 ** i) > 1 else 1 for i in range(depth)]
    else:
        raise ValueError(f"Unknown pattern '{pattern}'")


def experiment_width_grid_search(
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    base_widths: List[int],
    depth: int,
    patterns: List[str],
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.0,
    batch_norm: bool = False
):
    """
    Grid search по ширине: base_width и pattern.

    Args:
        base_widths: список базовых ширин для генерации слоев
        depth: число слоев
        patterns: схемы изменения ширины
    """
    input_dim = train_ds[0][0].shape[0]
    output_dim = len(set(y for _, y in train_ds))

    # Разбиваем train на train/val
    val_len = int(len(train_ds) * 0.2)
    train_len = len(train_ds) - val_len
    train_subds, val_ds = random_split(train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results = []

    for pattern in patterns:
        for base_width in base_widths:
            try:
                widths = generate_width_patterns(base_width, depth, pattern)
            except ValueError as e:
                logger.warning(f"Skipping invalid pattern: {e}")
                continue

            logger.info(f"Training model with base_width={base_width}, pattern={pattern}, layers={widths}")
            model = MLP(input_dim, output_dim, widths, dropout, batch_norm).to(device)

            start_time = time.time()
            history = train_model(model, train_loader, val_loader, epochs, lr, device)
            elapsed = time.time() - start_time

            criterion = nn.CrossEntropyLoss()
            test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)

            n_params = count_parameters(model)

            logger.info(
                f"Pattern={pattern}, BaseWidth={base_width}: Test acc={test_acc:.4f}, "
                f"Params={n_params}, Time={elapsed:.2f}s"
            )

            results.append({
                'pattern': pattern,
                'base_width': base_width,
                'widths': widths,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'params': n_params,
                'time': elapsed
            })

    plot_grid_search_heatmap(results, base_widths, patterns)


def plot_grid_search_heatmap(
    results: List[dict],
    base_widths: List[int],
    patterns: List[str]
):
    """
    Визуализация результатов grid search в виде heatmap (accuracy).

    По оси X - base_width, по оси Y - pattern.
    """
    # Создаем матрицу accuracy
    acc_matrix = np.zeros((len(patterns), len(base_widths)))

    # Заполняем матрицу
    for r in results:
        i = patterns.index(r['pattern'])
        j = base_widths.index(r['base_width'])
        acc_matrix[i, j] = r['test_acc']

    plt.figure(figsize=(10, 6))
    sns.heatmap(acc_matrix, annot=True, fmt=".3f", xticklabels=base_widths, yticklabels=patterns, cmap="viridis")
    plt.xlabel("Base Width")
    plt.ylabel("Width Pattern")
    plt.title("Grid Search: Test Accuracy Heatmap")
    plt.show()


def test_simple_run():
    """
    Тестирование: запускаем короткий эксперимент с одной конфигурацией, 3 слоя, 5 эпох.
    """
    logger.info("Running simple test for width experiments...")
    train_ds, test_ds = load_and_prepare_data(test_size=0.5, random_state=0)

    layer_widths = {
        'test_config': [64, 32, 16]
    }
    experiment_width_comparison(
        train_ds, test_ds,
        layer_widths=layer_widths,
        epochs=5,
        batch_size=32,
        lr=1e-3,
        dropout=0.0,
        batch_norm=False
    )
    logger.info("Simple test passed.")


if __name__ == "__main__":
    # Тест
    test_simple_run()

    # Основные параметры
    train_ds, test_ds = load_and_prepare_data()
    epochs = 40
    batch_size = 64
    lr = 1e-3

    # 2.1 Сравнение моделей разной ширины
    layer_widths = {
        'narrow': [64, 32, 16],
        'medium': [256, 128, 64],
        'wide': [1024, 512, 256],
        'very_wide': [2048, 1024, 512]
    }
    logger.info("=== Experiment 2.1: Width Comparison ===")
    experiment_width_comparison(
        train_ds, test_ds,
        layer_widths=layer_widths,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        dropout=0.0,
        batch_norm=False
    )

    # 2.2 Оптимизация архитектуры
    base_widths = [16, 32, 64, 128, 256]
    depth = 3
    patterns = ['constant', 'widening', 'narrowing']

    logger.info("=== Experiment 2.2: Grid Search for Optimal Width Architecture ===")
    experiment_width_grid_search(
        train_ds, test_ds,
        base_widths=base_widths,
        depth=depth,
        patterns=patterns,
        epochs=30,
        batch_size=batch_size,
        lr=lr,
        dropout=0.0,
        batch_norm=False
    )
