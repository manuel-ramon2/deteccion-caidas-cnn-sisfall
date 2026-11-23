"""Arquitecturas comparadas.

- CasilariCNN: CNN 1D de referencia (Casilari, Lora-Rivera y García-Lagos, Sensors 2020,
  Tabla 5): 4 bloques conv(1x5, padding 2, stride 1) + batch norm + ReLU + max pooling 1x5
  (el último bloque sin pooling), filtros 16-32-64-128, capa densa y softmax.
- InceptionTime: red moderna para clasificación de series de tiempo
  (Ismail Fawaz et al., Data Min. Knowl. Disc. 2020): 6 módulos Inception con
  cuello de botella, núcleos 40/20/10, atajos residuales cada 3 módulos y
  promedio global.
"""
from __future__ import annotations

import torch
from torch import nn


class CasilariCNN(nn.Module):
    """CNN de referencia. Supuestos no especificados en el artículo:
    stride del pooling = 5 (pooling no solapado, 'down-sampling') y dropout 0,5
    antes de la capa densa.
    """

    def __init__(self, canales: int = 3, largo: int = 1001, clases: int = 2, dropout: float = 0.5):
        super().__init__()
        filtros = [16, 32, 64, 128]
        capas, c_in = [], canales
        for i, f in enumerate(filtros):
            capas += [nn.Conv1d(c_in, f, kernel_size=5, padding=2, stride=1),
                      nn.BatchNorm1d(f), nn.ReLU(inplace=True)]
            if i < len(filtros) - 1:
                capas.append(nn.MaxPool1d(kernel_size=5, stride=5))
            c_in = f
        self.extractor = nn.Sequential(*capas)
        with torch.no_grad():
            n = self.extractor(torch.zeros(1, canales, largo)).numel()
        self.clasificador = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n, clases))

    @property
    def capa_gradcam(self) -> nn.Module:
        return self.extractor[-1]  # ReLU del último bloque convolucional

    def forward(self, x):
        return self.clasificador(self.extractor(x))


class ModuloInception(nn.Module):
    def __init__(self, c_in: int, nf: int = 32, nucleos=(39, 19, 9), cuello: int = 32):
        super().__init__()
        usa_cuello = c_in > 1
        self.cuello = nn.Conv1d(c_in, cuello, 1, bias=False) if usa_cuello else nn.Identity()
        c_b = cuello if usa_cuello else c_in
        self.convs = nn.ModuleList([nn.Conv1d(c_b, nf, k, padding=k // 2, bias=False) for k in nucleos])
        self.rama_pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(c_in, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        b = self.cuello(x)
        y = torch.cat([c(b) for c in self.convs] + [self.rama_pool(x)], dim=1)
        return self.act(self.bn(y))


class InceptionTime(nn.Module):
    def __init__(self, canales: int = 3, clases: int = 2, nf: int = 32, profundidad: int = 6):
        super().__init__()
        self.modulos = nn.ModuleList()
        self.atajos = nn.ModuleList()
        c = canales
        c_res = canales
        for d in range(profundidad):
            self.modulos.append(ModuloInception(c, nf))
            c = nf * 4
            if d % 3 == 2:
                self.atajos.append(nn.Sequential(nn.Conv1d(c_res, c, 1, bias=False), nn.BatchNorm1d(c)))
                c_res = c
        self.act = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c, clases)
        self._ultimo = None

    def forward(self, x):
        res = x
        k = 0
        for d, m in enumerate(self.modulos):
            x = m(x)
            if d % 3 == 2:
                x = self.act(x + self.atajos[k](res))
                res = x
                k += 1
        self._ultimo = x
        return self.fc(self.gap(x).squeeze(-1))


def crear(nombre: str) -> nn.Module:
    if nombre == "CNN-Casilari":
        return CasilariCNN()
    if nombre == "InceptionTime":
        return InceptionTime()
    raise ValueError(nombre)


def contar_macs(modelo: nn.Module, entrada: torch.Tensor) -> int:
    """Multiplicaciones-acumulaciones de capas Conv1d y Linear para una muestra."""
    total = 0

    def gancho(m, i, o):
        nonlocal total
        if isinstance(m, nn.Conv1d):
            total += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0]
        elif isinstance(m, nn.Linear):
            total += m.in_features * m.out_features

    hs = [m.register_forward_hook(gancho) for m in modelo.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    with torch.no_grad():
        modelo(entrada)
    for h in hs:
        h.remove()
    return total
