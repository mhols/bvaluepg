# Leichtes, aber vollständiges Python
FROM python:3.11

# System-Tools & wissenschaftliche Libs (für numpy/scipy/polyagamma etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc gfortran \
    libopenblas-dev liblapack-dev \
    libffi-dev \
    git \
 && rm -rf /var/lib/apt/lists/*

# Matplotlib ohne Display (Server/CI-kompatibel)
ENV MPLBACKEND=Agg
WORKDIR /app

# 1) Requirements getrennt kopieren (Build-Cache nutzen)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 2) Projektcode kopieren
COPY . /app

# Standardbefehl: Hinweistext (überschreiben beim Run)
CMD ["python", "-c", "print('Container bereit. Starte z.B.: python photonenCount4.py')"]
