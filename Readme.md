# AI Abliteration

Dieses Projekt implementiert verschiedene Methoden der **AI Abliteration** für Large Language Models (LLMs) auf Basis
aktueller Forschungsarbeiten. Ziel ist es, die Auswirkungen gezielter Gewichtsmodifikationen auf das Verhalten eines
Sprachmodells zu untersuchen und unterschiedliche Abliterationsmethoden miteinander zu vergleichen.

Die Implementierung unterstützt verschiedene Qwen-Modelle über die Hugging Face Transformers-Bibliothek und führt eine
vollständige Abliterationspipeline aus – von der Berechnung der Interventionsrichtung über die Modifikation der
Modellgewichte bis zur Evaluation der erzeugten Antworten.

---

## Features

- Implementierung verschiedener Abliterationsmethoden
    - Standard Abliteration
    - Norm Preserving Abliteration
- Berechnung der Refusal Direction aus Aktivierungen
- GPU-beschleunigte Ausführung mit PyTorch
- Unterstützung verschiedener Qwen-Modelle über Hugging Face
- Automatische Evaluation der Modellantworten

---

## Voraussetzungen

Folgende Software wird benötigt:

- Python **3.11**
- CUDA-fähige NVIDIA GPU (empfohlen)
- Git

Folgende Hardware wird benötigt:

- GPU
    - 3B-Modelle benötigt 8 GB VRAM.
    - 7B-Modelle benötigt 16 GB VRAM.
- CPU
    - 3B-Modelle benötigt 16 GB RAM.
    - 7B-Modelle benötigt 32 GB RAM.

---

## Installation

### Repository klonen

```bash
git clone <repository-url>
cd <repository-name>
```

### Virtuelle Umgebung erstellen

Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Python-Version prüfen

```bash
python --version
```

Es sollte mindestens folgende Version angezeigt werden:

```text
Python 3.11.x
```

### Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

---

## Hugging Face

Für den Download der Modelle wird ein Hugging Face Account benötigt.

Erstelle zunächst einen persönlichen Access Token (https://huggingface.co/settings/tokens) auf der Hugging Face Website
und hinterlege diesen anschließend als Umgebungsvariable `HF_TOKEN`.

### Windows (PowerShell)

```powershell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Windows (dauerhaft)

```powershell
setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

Anschließend das Terminal neu starten.

### Linux / macOS

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

Für eine dauerhafte Konfiguration kann die Zeile beispielsweise in die Datei `~/.bashrc` oder `~/.zshrc` eingetragen
werden.

Beim Start der Pipeline wird der Access Token automatisch über die Umgebungsvariable erkannt und für die
Authentifizierung bei Hugging Face verwendet. Ein manueller Login mit `huggingface-cli login` ist daher nicht
erforderlich.

Die heruntergeladenen Modelle werden anschließend automatisch im lokalen Hugging Face Cache gespeichert.

---

## Ausführung

Die Pipeline wird über die Datei

```text
run_pipeline.py
```

gestartet.

### Parameter

| Parameter  | Beschreibung                        |
|------------|-------------------------------------|
| `--model`  | Name des Hugging Face Modells       |
| `--method` | Zu verwendende Abliterationsmethode |

### Beispiel

```bash
python code/abliteration/run_pipeline.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --method norm_preserving
```

oder

```bash
python code/abliteration/run_pipeline.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --method standard
```

---

## Unterstützte Abliterationsmethoden

| Methode           | Beschreibung                                                               |
|-------------------|----------------------------------------------------------------------------|
| `standard`        | Entfernt die Projektion der Gewichte auf die Refusal Direction.            |
| `norm_preserving` | Entfernt die Projektion unter Erhaltung der ursprünglichen Gewichtsnormen. |

---

## Verwendete Technologien

- Python 3.11
- PyTorch
- Hugging Face Transformers
- CUDA
- NumPy
- Safetensors

---

## Literatur

Dieses Projekt basiert auf aktuellen Forschungsarbeiten zur AI Abliteration und Mechanistic Interpretability von Large
Language Models.

Unter anderem:

- AI Abliteration: A Comparative Study of Methods, Safety, and Performance
- Refusal in Language Models Is Mediated by a Single Direction
- Abliteration Mitigation via Refusal Aliases
- Hugging Face Transformers

---

## Lizenz

Dieses Projekt wurde im Rahmen einer Studienarbeit entwickelt und dient ausschließlich Forschungs- und Lehrzwecken.