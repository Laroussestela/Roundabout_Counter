# Roundabout_Counter

**RoundaboutCounter** es un sistema de visión por computador que detecta y cuenta vehículos que ingresan a una **rotonda** utilizando una cámara colocada en vista cenital. Emplea un modelo **YOLOv11** ajustado (fine-tuning) para detectar coches con mayor precisión desde esta perspectiva, y utiliza el algoritmo **SORT** para realizar un seguimiento en tiempo real.


<p align="center">
  <img src="https://github.com/user-attachments/assets/fe8b8a11-a19a-43e1-8dde-639528ce8f6d" width="45%" />
  <img src="https://github.com/user-attachments/assets/948bb12e-7e11-4da9-a09f-8d7b63a8d138" width="45%" />
</p>

## Requisitos

- Python 3.8 o superior
- OpenCV
- NumPy
- YOLOv11
- SORT (archivo local)

Instalación de dependencias:

```bash
pip install -r requirements.txt
